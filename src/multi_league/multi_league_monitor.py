"""
Multi-League Performance Monitor.

Aggregates performance tracking across all leagues.
"""

import logging
from typing import Dict, List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.monitoring.performance_monitor import PerformanceMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiLeagueMonitor:
    """
    Monitor performance across multiple leagues.
    
    Each league has independent tracking, with
    aggregated views for overall system performance.
    """
    
    def __init__(self, league_ids: List[str], db_dir='data'):
        """
        Initialize multi-league monitor.
        
        Args:
            league_ids: List of league IDs to monitor
            db_dir: Directory for performance databases
        """
        self.league_ids = league_ids
        self.db_dir = Path(db_dir)
        
        # Create monitor for each league
        self.monitors = {}
        for league_id in league_ids:
            db_path = self.db_dir / f'performance_{league_id}.db'
            self.monitors[league_id] = PerformanceMonitor(str(db_path))
        
        logger.info(f"✓ Multi-league monitor initialized ({len(league_ids)} leagues)")
    
    def get_league_summary(self, league_id: str, period='all') -> Dict:
        """Get summary for one league."""
        if league_id not in self.monitors:
            raise ValueError(f"Unknown league: {league_id}")
        
        monitor = self.monitors[league_id]
        summary = monitor.get_performance_summary(period)
        summary['league_id'] = league_id
        
        return summary
    
    def get_all_leagues_summary(self, period='all') -> Dict:
        """Get summary for all leagues."""
        summaries = {}
        
        for league_id in self.league_ids:
            summaries[league_id] = self.get_league_summary(league_id, period)
        
        return summaries
    
    def get_overall_summary(self, period='all') -> Dict:
        """Get aggregated summary across all leagues."""
        league_summaries = self.get_all_leagues_summary(period)
        
        # Aggregate totals
        total_bets = sum(s['total_bets'] for s in league_summaries.values())
        total_wins = sum(s['wins'] for s in league_summaries.values())
        total_profit = sum(s['total_profit'] for s in league_summaries.values())
        
        # Weighted averages
        if total_bets > 0:
            avg_roi = sum(
                s['roi'] * s['total_bets'] 
                for s in league_summaries.values()
            ) / total_bets
            
            avg_clv = sum(
                s['clv'] * s['total_bets']
                for s in league_summaries.values()
                if s['total_bets'] > 0
            ) / total_bets if total_bets > 0 else 0
        else:
            avg_roi = 0
            avg_clv = 0
        
        return {
            'total_bets': total_bets,
            'total_wins': total_wins,
            'hit_rate': (total_wins / total_bets * 100) if total_bets > 0 else 0,
            'total_profit': total_profit,
            'avg_roi': avg_roi,
            'avg_clv': avg_clv,
            'by_league': league_summaries
        }
    
    def generate_multi_league_report(self, period='all') -> str:
        """Generate formatted report for all leagues."""
        overall = self.get_overall_summary(period)
        
        report = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MULTI-LEAGUE PERFORMANCE REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OVERALL:
  Total Bets: {overall['total_bets']}
  Wins: {overall['total_wins']} ({overall['hit_rate']:.1f}%)
  Total Profit: £{overall['total_profit']:+.2f}
  Avg ROI: {overall['avg_roi']:+.1f}%
  Avg CLV: {overall['avg_clv']:+.1f}%

BY LEAGUE:
"""
        
        for league_id, summary in overall['by_league'].items():
            report += f"""
  {league_id.upper()}:
    Bets: {summary['total_bets']}
    Wins: {summary['wins']} ({summary['hit_rate']:.1f}%)
    Profit: £{summary['total_profit']:+.2f}
    ROI: {summary['roi']:+.1f}%
    CLV: {summary['clv']:+.1f}%
"""
        
        report += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        return report


def test_multi_league_monitor():
    """Test multi-league monitoring."""
    print("=" * 70)
    print("MULTI-LEAGUE MONITOR TEST")
    print("=" * 70)
    
    # Initialize monitor for 3 leagues
    monitor = MultiLeagueMonitor(['epl', 'laliga', 'bundesliga'])
    
    # Simulate some bets across leagues
    print("\n1. Simulating bets...")
    
    # EPL bets
    monitor.monitors['epl'].track_bet(
        'epl_001', 'City vs Pool', 'home', 2.10, 100, 'win', 5.0, 3.5
    )
    monitor.monitors['epl'].track_bet(
        'epl_002', 'Chelsea vs Arsenal', 'draw', 3.20, 50, 'loss', 4.5, -2.1
    )
    
    # La Liga bets
    monitor.monitors['laliga'].track_bet(
        'laliga_001', 'Real vs Barca', 'home', 2.20, 100, 'win', 6.0, 4.2
    )
    
    # Bundesliga bets
    monitor.monitors['bundesliga'].track_bet(
        'bund_001', 'Bayern vs Dortmund', 'home', 1.90, 100, 'loss', 3.0, -1.5
    )
    
    print("\n2. Overall Summary:")
    overall = monitor.get_overall_summary()
    print(f"  Total Bets: {overall['total_bets']}")
    print(f"  Total Profit: £{overall['total_profit']:+.2f}")
    print(f"  Avg ROI: {overall['avg_roi']:+.1f}%")
    
    print("\n3. Full Report:")
    report = monitor.generate_multi_league_report()
    print(report)
    
    print("✅ Multi-league monitor test complete!")


if __name__ == '__main__':
    test_multi_league_monitor()
