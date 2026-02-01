"""
Advanced Staking Module with Risk Management.

Implements:
1. Kelly Criterion staking (full and fractional)
2. Max exposure limits (per bet, per league, per day)
3. Drawdown tracking and protective limits

Usage:
    staker = Staker(bankroll=1000)
    stake = staker.calculate_stake(prob=0.45, odds=2.5, league='E0')
    staker.record_bet(stake=stake, result='win', odds=2.5)
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Staker:
    """
    Advanced staking with Kelly criterion and risk management.
    """
    
    # Configuration defaults
    DEFAULT_CONFIG = {
        'kelly_fraction': 0.25,          # Use 25% Kelly (safer)
        'max_stake_pct': 0.05,            # Max 5% of bankroll per bet
        'max_daily_exposure_pct': 0.20,   # Max 20% bankroll at risk per day
        'max_league_exposure_pct': 0.10,  # Max 10% on single league
        'min_stake': 1.0,                 # Minimum stake amount
        'drawdown_pause_pct': 0.25,       # Pause betting if 25% drawdown
        'drawdown_reduce_pct': 0.15,      # Reduce stakes at 15% drawdown
    }
    
    def __init__(
        self,
        bankroll: float = 1000.0,
        config: Optional[Dict] = None,
        state_file: Optional[str] = None
    ):
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.state_file = Path(state_file) if state_file else None
        
        # Tracking
        self.bet_history = []
        self.daily_exposure = defaultdict(float)  # date -> exposure
        self.league_exposure = defaultdict(float)  # league -> exposure
        self.peak_bankroll = bankroll
        
        # Load state if available
        if self.state_file and self.state_file.exists():
            self._load_state()
    
    def kelly_stake(
        self,
        prob: float,
        odds: float,
        fraction: Optional[float] = None
    ) -> float:
        """
        Calculate Kelly criterion stake.
        
        Args:
            prob: Model probability of winning
            odds: Decimal odds
            fraction: Kelly fraction (default from config)
            
        Returns:
            Stake amount
        """
        fraction = fraction or self.config['kelly_fraction']
        
        # Kelly formula: (bp - q) / b
        # where b = odds - 1, p = prob, q = 1 - p
        b = odds - 1
        p = prob
        q = 1 - p
        
        if b <= 0:
            return 0.0
        
        kelly = (b * p - q) / b
        
        # Apply fraction (fractional Kelly)
        kelly = kelly * fraction
        
        # Must be positive
        if kelly <= 0:
            return 0.0
        
        return self.bankroll * kelly
    
    def calculate_stake(
        self,
        prob: float,
        odds: float,
        league: str = 'unknown',
        apply_limits: bool = True
    ) -> float:
        """
        Calculate recommended stake with all risk limits.
        
        Args:
            prob: Model probability
            odds: Decimal odds
            league: League code for exposure tracking
            apply_limits: Whether to apply risk limits
            
        Returns:
            Final stake amount (may be 0 if limits exceeded)
        """
        # Base Kelly stake
        stake = self.kelly_stake(prob, odds)
        
        if stake <= 0:
            return 0.0
        
        if not apply_limits:
            return max(stake, self.config['min_stake'])
        
        # Apply limits
        stake = self._apply_limits(stake, league)
        
        # Check drawdown
        if self._check_drawdown_pause():
            logger.warning("Betting paused due to drawdown limit")
            return 0.0
        
        # Apply drawdown reduction if needed
        stake = self._apply_drawdown_reduction(stake)
        
        # Minimum stake
        if stake < self.config['min_stake']:
            return 0.0
        
        return round(stake, 2)
    
    def _apply_limits(self, stake: float, league: str) -> float:
        """Apply all stake limits."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # 1. Max stake per bet
        max_single = self.bankroll * self.config['max_stake_pct']
        stake = min(stake, max_single)
        
        # 2. Daily exposure limit
        daily_limit = self.bankroll * self.config['max_daily_exposure_pct']
        remaining_daily = daily_limit - self.daily_exposure[today]
        if remaining_daily <= 0:
            logger.warning("Daily exposure limit reached")
            return 0.0
        stake = min(stake, remaining_daily)
        
        # 3. League exposure limit
        league_limit = self.bankroll * self.config['max_league_exposure_pct']
        remaining_league = league_limit - self.league_exposure[league]
        if remaining_league <= 0:
            logger.warning(f"League exposure limit reached for {league}")
            return 0.0
        stake = min(stake, remaining_league)
        
        return stake
    
    def _check_drawdown_pause(self) -> bool:
        """Check if betting should pause due to drawdown."""
        if self.bankroll >= self.peak_bankroll:
            return False
        
        drawdown = (self.peak_bankroll - self.bankroll) / self.peak_bankroll
        return drawdown >= self.config['drawdown_pause_pct']
    
    def _apply_drawdown_reduction(self, stake: float) -> float:
        """Reduce stakes during minor drawdown."""
        if self.bankroll >= self.peak_bankroll:
            return stake
        
        drawdown = (self.peak_bankroll - self.bankroll) / self.peak_bankroll
        
        if drawdown >= self.config['drawdown_reduce_pct']:
            # Reduce by half during moderate drawdown
            reduction = 0.5
            logger.info(f"Reducing stake by {reduction:.0%} due to {drawdown:.1%} drawdown")
            return stake * reduction
        
        return stake
    
    def record_bet(
        self,
        stake: float,
        result: str,  # 'win', 'loss', 'void'
        odds: float,
        league: str = 'unknown'
    ):
        """
        Record bet outcome and update state.
        
        Args:
            stake: Amount staked
            result: 'win', 'loss', or 'void'
            odds: Decimal odds
            league: League code
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Calculate profit/loss
        if result == 'win':
            profit = stake * (odds - 1)
        elif result == 'loss':
            profit = -stake
        else:  # void
            profit = 0.0
        
        # Update bankroll
        self.bankroll += profit
        
        # Update peak
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll
        
        # Update exposure (only track open bets, reset on result)
        # For simplicity, we decrease exposure when bet settles
        self.daily_exposure[today] = max(0, self.daily_exposure[today] - stake)
        self.league_exposure[league] = max(0, self.league_exposure[league] - stake)
        
        # Record
        self.bet_history.append({
            'timestamp': datetime.now().isoformat(),
            'stake': stake,
            'odds': odds,
            'result': result,
            'profit': profit,
            'bankroll': self.bankroll,
            'league': league
        })
        
        # Save state
        self._save_state()
        
        logger.info(
            f"Bet recorded: {result} | Stake: {stake:.2f} | "
            f"Profit: {profit:+.2f} | Bankroll: {self.bankroll:.2f}"
        )
    
    def add_pending_bet(self, stake: float, league: str = 'unknown'):
        """Track pending bet exposure."""
        today = datetime.now().strftime('%Y-%m-%d')
        self.daily_exposure[today] += stake
        self.league_exposure[league] += stake
    
    def get_stats(self) -> Dict:
        """Get betting statistics."""
        if not self.bet_history:
            return {
                'total_bets': 0,
                'bankroll': self.bankroll,
                'roi': 0.0
            }
        
        wins = sum(1 for b in self.bet_history if b['result'] == 'win')
        losses = sum(1 for b in self.bet_history if b['result'] == 'loss')
        total_staked = sum(b['stake'] for b in self.bet_history if b['result'] != 'void')
        total_profit = sum(b['profit'] for b in self.bet_history)
        
        roi = total_profit / total_staked if total_staked > 0 else 0
        
        return {
            'total_bets': len(self.bet_history),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
            'total_staked': total_staked,
            'total_profit': total_profit,
            'roi': roi,
            'bankroll': self.bankroll,
            'peak_bankroll': self.peak_bankroll,
            'drawdown': (self.peak_bankroll - self.bankroll) / self.peak_bankroll if self.peak_bankroll > 0 else 0
        }
    
    def _save_state(self):
        """Persist state to file."""
        if not self.state_file:
            return
        
        state = {
            'bankroll': self.bankroll,
            'peak_bankroll': self.peak_bankroll,
            'initial_bankroll': self.initial_bankroll,
            'daily_exposure': dict(self.daily_exposure),
            'league_exposure': dict(self.league_exposure),
            'bet_history': self.bet_history[-100:],  # Keep last 100
            'last_updated': datetime.now().isoformat()
        }
        
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load state from file."""
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            self.bankroll = state.get('bankroll', self.bankroll)
            self.peak_bankroll = state.get('peak_bankroll', self.peak_bankroll)
            self.initial_bankroll = state.get('initial_bankroll', self.initial_bankroll)
            self.daily_exposure = defaultdict(float, state.get('daily_exposure', {}))
            self.league_exposure = defaultdict(float, state.get('league_exposure', {}))
            self.bet_history = state.get('bet_history', [])
            
            logger.info(f"Loaded staker state: bankroll={self.bankroll:.2f}")
        except Exception as e:
            logger.warning(f"Failed to load staker state: {e}")


def kelly_simple(prob: float, odds: float, fraction: float = 0.25) -> float:
    """
    Simple Kelly stake calculation (standalone function).
    
    Args:
        prob: Probability of winning
        odds: Decimal odds
        fraction: Kelly fraction (default 25%)
        
    Returns:
        Fraction of bankroll to stake
    """
    b = odds - 1
    if b <= 0:
        return 0.0
    
    kelly = (b * prob - (1 - prob)) / b
    return max(0, kelly * fraction)


if __name__ == '__main__':
    # Test the staker
    staker = Staker(bankroll=1000)
    
    print("=" * 50)
    print("STAKING MODULE TEST")
    print("=" * 50)
    
    # Test scenarios
    test_cases = [
        {'prob': 0.45, 'odds': 2.5, 'expected': 'moderate stake'},
        {'prob': 0.60, 'odds': 2.0, 'expected': 'good value'},
        {'prob': 0.30, 'odds': 2.0, 'expected': 'no bet'},
        {'prob': 0.55, 'odds': 1.8, 'expected': 'small edge'},
    ]
    
    for tc in test_cases:
        stake = staker.calculate_stake(tc['prob'], tc['odds'], 'E0')
        kelly = kelly_simple(tc['prob'], tc['odds'])
        print(f"\nProb: {tc['prob']:.0%}, Odds: {tc['odds']}")
        print(f"  Kelly fraction: {kelly:.2%}")
        print(f"  Stake: ${stake:.2f} ({tc['expected']})")
    
    # Simulate some bets
    print("\n" + "=" * 50)
    print("SIMULATING BETS")
    print("=" * 50)
    
    import random
    random.seed(42)
    
    for i in range(10):
        prob = random.uniform(0.35, 0.55)
        odds = random.uniform(1.8, 3.0)
        stake = staker.calculate_stake(prob, odds, 'E0')
        
        if stake > 0:
            staker.add_pending_bet(stake, 'E0')
            # Simulate outcome
            won = random.random() < prob
            staker.record_bet(stake, 'win' if won else 'loss', odds, 'E0')
    
    # Print stats
    print("\n" + "=" * 50)
    print("FINAL STATS")
    print("=" * 50)
    stats = staker.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
