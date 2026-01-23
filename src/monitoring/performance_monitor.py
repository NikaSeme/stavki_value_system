"""
Performance monitoring and tracking.

Tracks all bets and calculates:
- ROI (Return on Investment)
- Yield percentage
- CLV (Closing Line Value)
- Hit rate / Win rate
- Maximum drawdown
"""

import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Track and analyze betting performance."""
    
    def __init__(self, db_path='data/performance.db'):
        """Initialize performance monitor."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path)
        self._init_db()
        
        logger.info(f"✓ Performance monitor initialized: {db_path}")
    
    def _init_db(self):
        """Create database schema."""
        cursor = self.conn.cursor()
        
        # Bets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bets (
                bet_id TEXT PRIMARY KEY,
                match_id TEXT,
                match_name TEXT,
                outcome TEXT,
                odds REAL,
                stake REAL,
                result TEXT,
                profit REAL,
                timestamp INTEGER,
                ev_percent REAL,
                clv_percent REAL
            )
        ''')
        
        # Performance snapshots
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                total_bets INTEGER,
                total_staked REAL,
                total_profit REAL,
                roi_percent REAL,
                hit_rate REAL,
                avg_clv REAL,
                max_drawdown REAL
            )
        ''')
        
        self.conn.commit()
        logger.info("✓ Database schema initialized")
    
    def track_bet(
        self,
        bet_id: str,
        match_name: str,
        outcome: str,
        odds: float,
        stake: float,
        result: str,
        ev_percent: float = 0.0,
        clv_percent: float = 0.0
    ):
        """
        Record bet outcome.
        
        Args:
            bet_id: Unique bet identifier
            match_name: Match description
            outcome: Bet outcome ('home', 'draw', 'away')
            odds: Bet odds
            stake: Amount staked
            result: 'win', 'loss', or 'push'
            ev_percent: Expected value percentage
            clv_percent: Closing line value percentage
        """
        # Calculate profit
        if result == 'win':
            profit = stake * (odds - 1)
        elif result == 'push':
            profit = 0
        else:  # loss
            profit = -stake
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO bets
            (bet_id, match_name, outcome, odds, stake, result, profit, timestamp, ev_percent, clv_percent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (bet_id, match_name, outcome, odds, stake, result, profit, int(time.time()), ev_percent, clv_percent))
        
        self.conn.commit()
        logger.info(f"✓ Tracked bet: {bet_id} ({result}, {profit:+.2f})")
    
    def get_bet_count(self, period='all') -> int:
        """Get number of bets."""
        query = "SELECT COUNT(*) FROM bets"
        if period != 'all':
            query += f" WHERE timestamp > {self._get_period_start(period)}"
        
        cursor = self.conn.cursor()
        return cursor.execute(query).fetchone()[0]
    
    def get_wins(self, period='all') -> int:
        """Get number of winning bets."""
        query = "SELECT COUNT(*) FROM bets WHERE result = 'win'"
        if period != 'all':
            query += f" AND timestamp > {self._get_period_start(period)}"
        
        cursor = self.conn.cursor()
        return cursor.execute(query).fetchone()[0]
    
    def get_profit(self, period='all') -> float:
        """Get total profit."""
        query = "SELECT SUM(profit) FROM bets"
        if period != 'all':
            query += f" WHERE timestamp > {self._get_period_start(period)}"
        
        cursor = self.conn.cursor()
        result = cursor.execute(query).fetchone()[0]
        return float(result) if result else 0.0
    
    def calculate_roi(self, period='all') -> float:
        """
        Calculate ROI (Return on Investment).
        
        ROI% = (Total Profit / Total Staked) * 100
        """
        query = "SELECT SUM(profit), SUM(stake) FROM bets"
        if period != 'all':
            query += f" WHERE timestamp > {self._get_period_start(period)}"
        
        cursor = self.conn.cursor()
        profit, staked = cursor.execute(query).fetchone()
        
        if not staked or staked == 0:
            return 0.0
        
        return (float(profit) / float(staked)) * 100
    
    def calculate_hit_rate(self, period='all') -> float:
        """Calculate win rate percentage."""
        total = self.get_bet_count(period)
        wins = self.get_wins(period)
        
        if total == 0:
            return 0.0
        
        return (wins / total) * 100
    
    def calculate_clv(self, period='all') -> float:
        """Calculate average CLV."""
        query = "SELECT AVG(clv_percent) FROM bets WHERE clv_percent IS NOT NULL"
        if period != 'all':
            query += f" AND timestamp > {self._get_period_start(period)}"
        
        cursor = self.conn.cursor()
        result = cursor.execute(query).fetchone()[0]
        return float(result) if result else 0.0
    
    def calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown.
        
        Drawdown = largest peak-to-trough decline in cumulative profit
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT profit FROM bets ORDER BY timestamp")
        
        profits = [row[0] for row in cursor.fetchall()]
        
        if not profits:
            return 0.0
        
        # Calculate cumulative profits
        cumulative = []
        total = 0
        for profit in profits:
            total += profit
            cumulative.append(total)
        
        # Find max drawdown
        peak = cumulative[0]
        max_dd = 0
        
        for value in cumulative:
            if value > peak:
                peak = value
            
            drawdown = peak - value
            if drawdown > max_dd:
                max_dd = drawdown
        
        return float(max_dd)
    
    def get_recent_record(self, n=10) -> str:
        """Get record of last n bets (e.g., '6W-4L')."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT result FROM bets 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (n,))
        
        results = [row[0] for row in cursor.fetchall()]
        
        wins = sum(1 for r in results if r == 'win')
        losses = sum(1 for r in results if r == 'loss')
        pushes = sum(1 for r in results if r == 'push')
        
        if pushes > 0:
            return f"{wins}W-{losses}L-{pushes}P"
        return f"{wins}W-{losses}L"
    
    def get_performance_summary(self, period='all') -> Dict:
        """Get comprehensive performance summary."""
        return {
            'total_bets': self.get_bet_count(period),
            'wins': self.get_wins(period),
            'hit_rate': self.calculate_hit_rate(period),
            'total_profit': self.get_profit(period),
            'roi': self.calculate_roi(period),
            'clv': self.calculate_clv(period),
            'max_drawdown': self.calculate_max_drawdown(),
            'recent_record': self.get_recent_record()
        }
    
    def save_snapshot(self):
        """Save current performance snapshot."""
        summary = self.get_performance_summary()
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO performance_snapshots
            (timestamp, total_bets, total_staked, total_profit, roi_percent, hit_rate, avg_clv, max_drawdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            int(time.time()),
            summary['total_bets'],
            0,  # total_staked (would need to query)
            summary['total_profit'],
            summary['roi'],
            summary['hit_rate'],
            summary['clv'],
            summary['max_drawdown']
        ))
        
        self.conn.commit()
    
    def _get_period_start(self, period: str) -> int:
        """Get Unix timestamp for period start."""
        now = time.time()
        
        if period == 'today':
            # Midnight today
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            return int(today.timestamp())
        elif period == 'week':
            return int(now - 7 * 24 * 3600)
        elif period == 'month':
            return int(now - 30 * 24 * 3600)
        
        return 0


def test_performance_monitor():
    """Test performance monitoring."""
    print("=" * 70)
    print("PERFORMANCE MONITOR TEST")
    print("=" * 70)
    
    monitor = PerformanceMonitor(db_path='data/test_performance.db')
    
    # Simulate some bets
    print("\n1. Simulating bets...")
    
    bets = [
        ('bet_001', 'City vs Pool', 'home', 2.10, 100, 'win', 5.0, 3.5),
        ('bet_002', 'Chelsea vs Arsenal', 'draw', 3.20, 50, 'loss', 4.5, -2.1),
        ('bet_003', 'Spurs vs United', 'away', 2.50, 100, 'win', 6.0, 5.2),
        ('bet_004', 'Brighton vs Villa', 'home', 1.90, 100, 'loss', 3.0, -1.5),
        ('bet_005', 'Newcastle vs West Ham', 'home', 2.00, 100, 'win', 5.5, 4.0),
    ]
    
    for bet in bets:
        monitor.track_bet(*bet)
    
    # Get summary
    print("\n2. Performance Summary:")
    summary = monitor.get_performance_summary()
    
    print(f"  Total Bets: {summary['total_bets']}")
    print(f"  Wins: {summary['wins']} ({summary['hit_rate']:.1f}%)")
    print(f"  Total Profit: £{summary['total_profit']:+.2f}")
    print(f"  ROI: {summary['roi']:+.1f}%")
    print(f"  Avg CLV: {summary['clv']:+.1f}%")
    print(f"  Max Drawdown: £{summary['max_drawdown']:.2f}")
    print(f"  Recent Record: {summary['recent_record']}")
    
    print("\n✅ Performance monitoring test complete!")


if __name__ == '__main__':
    test_performance_monitor()
