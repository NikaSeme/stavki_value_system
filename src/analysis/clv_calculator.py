"""
CLV (Closing Line Value) calculator and analyzer.

Tracks betting performance against closing lines.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

from ..data.odds_tracker import OddsTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLVCalculator:
    """
    Calculate and track Closing Line Value (CLV).
    
    CLV is the industry-standard metric for measuring bet quality.
    Positive CLV = beat the closing line (good)
    Negative CLV = worse than closing (bad)
    """
    
    def __init__(self, tracker: Optional[OddsTracker] = None):
        """Initialize CLV calculator."""
        self.tracker = tracker or OddsTracker()
    
    def calculate_clv(self, bet_odds: float, closing_odds: float) -> float:
        """
        Calculate CLV percentage.
        
        CLV% = (bet_odds - closing_odds) / closing_odds * 100
        
        Args:
            bet_odds: Odds you got when betting
            closing_odds: Final closing odds
            
        Returns:
            CLV percentage (positive = good)
        """
        if closing_odds == 0:
            return 0.0
        
        return ((bet_odds - closing_odds) / closing_odds) * 100
    
    def track_bet(
        self,
        bet_id: str,
        match_id: str,
        outcome: str,
        bet_odds: float,
        stake: float = 100.0
    ):
        """Record a bet for CLV tracking."""
        self.tracker.track_bet_for_clv(bet_id, match_id, outcome, bet_odds)
        
        logger.info(f"Tracking bet: {bet_id} @ {bet_odds:.2f} on {outcome}")
    
    def get_clv(self, bet_id: str) -> Optional[Dict]:
        """
        Get CLV for a specific bet.
        
        Returns:
            Dict with bet details and CLV, or None if not ready
        """
        clv_pct = self.tracker.calculate_clv(bet_id)
        
        if clv_pct is None:
            return None
        
        # Get bet details
        conn = sqlite3.connect(self.tracker.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT match_id, outcome, bet_odds, closing_odds
            FROM clv_tracking
            WHERE bet_id = ?
        ''', (bet_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        match_id, outcome, bet_odds, closing_odds = row
        
        return {
            'bet_id': bet_id,
            'match_id': match_id,
            'outcome': outcome,
            'bet_odds': bet_odds,
            'closing_odds': closing_odds,
            'clv_percent': clv_pct,
            'beat_closing': clv_pct > 0
        }
    
    def get_cumulative_clv(self) -> Dict:
        """
        Get cumulative CLV across all bets.
        
        Returns:
            Summary statistics
        """
        conn = sqlite3.connect(self.tracker.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_bets,
                AVG(clv_percent) as avg_clv,
                MIN(clv_percent) as min_clv,
                MAX(clv_percent) as max_clv,
                SUM(CASE WHEN clv_percent > 0 THEN 1 ELSE 0 END) as positive_clv_count
            FROM clv_tracking
            WHERE clv_percent IS NOT NULL
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if not row or row[0] == 0:
            return {
                'total_bets': 0,
                'avg_clv': 0.0,
                'positive_clv_rate': 0.0
            }
        
        total, avg, min_clv, max_clv, positive = row
        
        return {
            'total_bets': total,
            'avg_clv': float(avg) if avg else 0.0,
            'min_clv': float(min_clv) if min_clv else 0.0,
            'max_clv': float(max_clv) if max_clv else 0.0,
            'positive_clv_count': positive,
            'positive_clv_rate': (positive / total) * 100 if total > 0 else 0.0
        }
    
    def generate_report(self) -> str:
        """Generate CLV report."""
        stats = self.get_cumulative_clv()
        
        report = "=" * 60 + "\n"
        report += "CLV (CLOSING LINE VALUE) REPORT\n"
        report += "=" * 60 + "\n\n"
        
        if stats['total_bets'] == 0:
            report += "No bets tracked yet.\n"
            return report
        
        report += f"Total Bets: {stats['total_bets']}\n"
        report += f"Average CLV: {stats['avg_clv']:+.2f}%\n"
        report += f"Best CLV: {stats['max_clv']:+.2f}%\n"
        report += f"Worst CLV: {stats['min_clv']:+.2f}%\n"
        report += f"Positive CLV Rate: {stats['positive_clv_rate']:.1f}%\n"
        report += "\n"
        
        # Interpretation
        if stats['avg_clv'] > 2:
            report += "✅ EXCELLENT: Consistently beating closing lines!\n"
        elif stats['avg_clv'] > 0:
            report += "✓ GOOD: Positive CLV indicates edge\n"
        else:
            report += "⚠️  NEGATIVE: Losing to closing lines on average\n"
        
        report += "=" * 60 + "\n"
        
        return report


def test_clv():
    """Test CLV calculator."""
    print("=" * 60)
    print("CLV CALCULATOR TEST")
    print("=" * 60)
    
    # Create calculator
    calculator = CLVCalculator()
    
    # Setup test match
    tracker = calculator.tracker
    match_id = "clv_test_001"
    
    # Store closing line
    closing_odds = {'home': 2.00, 'draw': 3.20, 'away': 3.50}
    tracker.store_closing_line(
        match_id,
        "Team A",
        "Team B",
        1234567890,
        {'home': 2.10, 'draw': 3.30, 'away': 3.40},  # Opening
        closing_odds
    )
    
    # Track bets
    print("\n1. Tracking bets...")
    
    # Good bet (beat closing)
    calculator.track_bet("bet_good", match_id, 'home', 2.15, stake=100)
    
    # Bad bet (worse than closing)
    calculator.track_bet("bet_bad", match_id, 'away', 3.30, stake=100)
    
    # Calculate CLVs
    print("\n2. Calculating CLVs...")
    
    good_clv = calculator.get_clv('bet_good')
    bad_clv = calculator.get_clv('bet_bad')
    
    print(f"\nGood Bet:")
    print(f"  Bet odds: {good_clv['bet_odds']:.2f}")
    print(f"  Closing: {good_clv['closing_odds']:.2f}")
    print(f"  CLV: {good_clv['clv_percent']:+.2f}%")
    print(f"  Beat closing: {good_clv['beat_closing']}")
    
    print(f"\nBad Bet:")
    print(f"  Bet odds: {bad_clv['bet_odds']:.2f}")
    print(f"  Closing: {bad_clv['closing_odds']:.2f}")
    print(f"  CLV: {bad_clv['clv_percent']:+.2f}%")
    print(f"  Beat closing: {bad_clv['beat_closing']}")
    
    # Generate report
    print("\n3. Generating report...")
    print("\n" + calculator.generate_report())
    
    # Validate
    assert good_clv['clv_percent'] > 0, "Good bet should have positive CLV"
    assert bad_clv['clv_percent'] < 0, "Bad bet should have negative CLV"
    
    print("✅ All tests passed!")


if __name__ == '__main__':
    test_clv()
