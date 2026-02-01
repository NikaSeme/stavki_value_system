"""
Closing Line Value (CLV) Tracker.

CLV is the #1 metric for long-term profitability.
If you consistently beat the closing line, you're +EV
even if short-term variance is against you.

CLV = (1/bet_odds) - (1/closing_odds)
Positive CLV = you beat the market.

This module:
1. Stores bet placements with opening and closing odds
2. Calculates CLV for each bet
3. Generates reports by league, time, odds range
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLVTracker:
    """
    Track Closing Line Value for all bets.
    """
    
    def __init__(self, storage_file: str = 'data/clv_tracker.json'):
        self.storage_file = Path(storage_file)
        self.bets: List[Dict] = []
        self._load()
    
    def record_bet(
        self,
        match_id: str,
        outcome: str,  # 'home', 'draw', 'away'
        bet_odds: float,
        bet_time: Optional[datetime] = None,
        league: str = 'unknown',
        stake: float = 0.0
    ):
        """
        Record a bet placement.
        
        Call update_closing_odds() after match starts to complete CLV calculation.
        """
        bet = {
            'match_id': match_id,
            'outcome': outcome,
            'bet_odds': bet_odds,
            'bet_time': (bet_time or datetime.now()).isoformat(),
            'league': league,
            'stake': stake,
            'closing_odds': None,
            'clv': None,
            'result': None,  # 'win', 'loss', 'void'
            'profit': None
        }
        self.bets.append(bet)
        self._save()
        
        logger.info(f"Recorded bet: {match_id} {outcome} @ {bet_odds:.2f}")
    
    def update_closing_odds(
        self,
        match_id: str,
        outcome: str,
        closing_odds: float
    ):
        """
        Update closing odds for a bet and calculate CLV.
        
        Call this when the match starts (closing line is set).
        """
        for bet in self.bets:
            if bet['match_id'] == match_id and bet['outcome'] == outcome:
                if bet['closing_odds'] is None:
                    bet['closing_odds'] = closing_odds
                    bet['clv'] = self._calculate_clv(bet['bet_odds'], closing_odds)
                    self._save()
                    logger.info(
                        f"CLV for {match_id}: {bet['clv']:+.2%} "
                        f"(bet @ {bet['bet_odds']:.2f}, close @ {closing_odds:.2f})"
                    )
    
    def update_result(
        self,
        match_id: str,
        outcome: str,
        result: str,  # 'win', 'loss', 'void'
        profit: Optional[float] = None
    ):
        """Update bet result after match completes."""
        for bet in self.bets:
            if bet['match_id'] == match_id and bet['outcome'] == outcome:
                bet['result'] = result
                if profit is not None:
                    bet['profit'] = profit
                elif result == 'win' and bet['stake'] > 0:
                    bet['profit'] = bet['stake'] * (bet['bet_odds'] - 1)
                elif result == 'loss' and bet['stake'] > 0:
                    bet['profit'] = -bet['stake']
                else:
                    bet['profit'] = 0.0
                self._save()
    
    @staticmethod
    def _calculate_clv(bet_odds: float, closing_odds: float) -> float:
        """
        Calculate CLV.
        
        CLV = (1/bet_odds) - (1/closing_odds)
        Positive = you got better odds than closing
        """
        if bet_odds <= 1 or closing_odds <= 1:
            return 0.0
        return (1 / bet_odds) - (1 / closing_odds)
    
    def get_report(self, league: Optional[str] = None) -> Dict:
        """
        Generate CLV report.
        
        Returns:
            Dict with CLV statistics
        """
        bets = [b for b in self.bets if b['clv'] is not None]
        if league:
            bets = [b for b in bets if b['league'] == league]
        
        if not bets:
            return {'total_bets': 0, 'avg_clv': None}
        
        clvs = [b['clv'] for b in bets]
        positive_clvs = [c for c in clvs if c > 0]
        
        return {
            'total_bets': len(bets),
            'avg_clv': sum(clvs) / len(clvs),
            'positive_clv_rate': len(positive_clvs) / len(clvs),
            'total_clv': sum(clvs),
            'best_clv': max(clvs),
            'worst_clv': min(clvs),
        }
    
    def get_league_breakdown(self) -> Dict[str, Dict]:
        """Get CLV breakdown by league."""
        leagues = set(b['league'] for b in self.bets if b['clv'] is not None)
        return {league: self.get_report(league) for league in leagues}
    
    def get_profit_correlation(self) -> Optional[float]:
        """
        Calculate correlation between CLV and actual profit.
        
        High correlation = CLV is a good predictor of profitability.
        """
        bets = [b for b in self.bets if b['clv'] is not None and b['profit'] is not None]
        if len(bets) < 10:
            return None
        
        import numpy as np
        clvs = np.array([b['clv'] for b in bets])
        profits = np.array([b['profit'] for b in bets])
        
        # Normalize profits to 0/1 win/loss
        profits_binary = (profits > 0).astype(float)
        
        correlation = np.corrcoef(clvs, profits_binary)[0, 1]
        return correlation
    
    def _load(self):
        """Load bets from storage."""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'r') as f:
                    self.bets = json.load(f)
                logger.info(f"Loaded {len(self.bets)} CLV records")
            except Exception as e:
                logger.warning(f"Failed to load CLV data: {e}")
                self.bets = []
    
    def _save(self):
        """Save bets to storage."""
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_file, 'w') as f:
            json.dump(self.bets, f, indent=2)


def calculate_simple_clv(bet_odds: float, closing_odds: float) -> float:
    """
    Simple CLV calculation (standalone function).
    
    Args:
        bet_odds: Odds at time of bet
        closing_odds: Odds at kickoff
        
    Returns:
        CLV as decimal (positive = beat the line)
    """
    if bet_odds <= 1 or closing_odds <= 1:
        return 0.0
    return (1 / bet_odds) - (1 / closing_odds)


if __name__ == '__main__':
    # Test CLV tracker
    import tempfile
    
    print("=" * 50)
    print("CLV TRACKER TEST")
    print("=" * 50)
    
    # Use temp file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name
    
    tracker = CLVTracker(storage_file=temp_path)
    
    # Simulate bets
    test_bets = [
        {'mid': 'match1', 'outcome': 'home', 'bet': 2.10, 'close': 2.00, 'result': 'win'},
        {'mid': 'match2', 'outcome': 'away', 'bet': 3.50, 'close': 3.80, 'result': 'loss'},
        {'mid': 'match3', 'outcome': 'draw', 'bet': 3.20, 'close': 3.10, 'result': 'loss'},
        {'mid': 'match4', 'outcome': 'home', 'bet': 1.90, 'close': 1.85, 'result': 'win'},
    ]
    
    for tb in test_bets:
        tracker.record_bet(
            match_id=tb['mid'],
            outcome=tb['outcome'],
            bet_odds=tb['bet'],
            league='E0',
            stake=10.0
        )
        tracker.update_closing_odds(tb['mid'], tb['outcome'], tb['close'])
        tracker.update_result(tb['mid'], tb['outcome'], tb['result'])
    
    # Report
    print("\n" + "-" * 50)
    print("CLV REPORT")
    print("-" * 50)
    
    report = tracker.get_report()
    for k, v in report.items():
        if v is not None and isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Cleanup
    Path(temp_path).unlink()
    print("\n✓ CLV Tracker working")
