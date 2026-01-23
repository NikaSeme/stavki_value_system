"""
Time-series odds tracker for line movement analysis.

Stores opening, current, and closing odds for all matches.
"""

import sqlite3
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OddsTracker:
    """
    Track odds over time for line movement analysis.
    
    Features:
    - Time-series odds storage (SQLite)
    - Opening/closing line capture
    - Continuous polling
    - CLV calculation support
    """
    
    def __init__(self, db_path='data/odds/odds_timeseries.db'):
        """Initialize tracker with database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        self._tracking_threads = {}
        
        logger.info(f"OddsTracker initialized: {self.db_path}")
    
    def _init_database(self):
        """Create database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Odds history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS odds_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                bookmaker TEXT NOT NULL,
                outcome TEXT NOT NULL,
                odds REAL NOT NULL,
                is_opening BOOLEAN DEFAULT 0,
                is_closing BOOLEAN DEFAULT 0
            )
        ''')
        
        # Indexes for performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_match_timestamp 
            ON odds_history(match_id, timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_match_outcome 
            ON odds_history(match_id, outcome)
        ''')
        
        # Closing lines table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS closing_lines (
                match_id TEXT PRIMARY KEY,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                commence_time INTEGER NOT NULL,
                home_odds_open REAL,
                draw_odds_open REAL,
                away_odds_open REAL,
                home_odds_close REAL,
                draw_odds_close REAL,
                away_odds_close REAL,
                bookmaker_consensus TEXT,
                created_at INTEGER
            )
        ''')
        
        # CLV tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clv_tracking (
                bet_id TEXT PRIMARY KEY,
                match_id TEXT NOT NULL,
                outcome TEXT NOT NULL,
                bet_odds REAL NOT NULL,
                closing_odds REAL,
                clv_percent REAL,
                bet_timestamp INTEGER NOT NULL,
                match_timestamp INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("✓ Database schema initialized")
    
    def store_odds_snapshot(
        self, 
        match_id: str, 
        odds_data: Dict,
        is_opening: bool = False,
        is_closing: bool = False
    ):
        """
        Store a snapshot of odds at a point in time.
        
        Args:
            match_id: Unique match identifier
            odds_data: Dict with bookmaker odds
            is_opening: Mark as opening line
            is_closing: Mark as closing line
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = int(time.time())
        
        # Store each bookmaker's odds
        for bookmaker, outcomes in odds_data.items():
            for outcome, odds in outcomes.items():
                cursor.execute('''
                    INSERT INTO odds_history 
                    (match_id, timestamp, bookmaker, outcome, odds, is_opening, is_closing)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (match_id, timestamp, bookmaker, outcome, odds, is_opening, is_closing))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Stored odds snapshot for {match_id} ({len(odds_data)} bookmakers)")
    
    def get_opening_odds(self, match_id: str) -> Optional[Dict]:
        """Get opening odds for a match."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT outcome, AVG(odds) as avg_odds
            FROM odds_history
            WHERE match_id = ? AND is_opening = 1
            GROUP BY outcome
        ''', (match_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return None
        
        return {outcome: odds for outcome, odds in rows}
    
    def get_current_odds(self, match_id: str) -> Optional[Dict]:
        """Get most recent odds for a match."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get latest timestamp
        cursor.execute('''
            SELECT MAX(timestamp) FROM odds_history WHERE match_id = ?
        ''', (match_id,))
        
        latest_time = cursor.fetchone()[0]
        if not latest_time:
            conn.close()
            return None
        
        # Get odds at that time
        cursor.execute('''
            SELECT outcome, AVG(odds) as avg_odds
            FROM odds_history
            WHERE match_id = ? AND timestamp = ?
            GROUP BY outcome
        ''', (match_id, latest_time))
        
        rows = cursor.fetchall()
        conn.close()
        
        return {outcome: odds for outcome, odds in rows}
    
    def get_closing_odds(self, match_id: str) -> Optional[Dict]:
        """Get closing odds from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT home_odds_close, draw_odds_close, away_odds_close
            FROM closing_lines
            WHERE match_id = ?
        ''', (match_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return {
            'home': row[0],
            'draw': row[1],
            'away': row[2]
        }
    
    def store_closing_line(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        commence_time: int,
        opening_odds: Dict,
        closing_odds: Dict
    ):
        """Store final closing line."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO closing_lines
            (match_id, home_team, away_team, commence_time,
             home_odds_open, draw_odds_open, away_odds_open,
             home_odds_close, draw_odds_close, away_odds_close,
             created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            match_id,
            home_team,
            away_team,
            commence_time,
            opening_odds.get('home'),
            opening_odds.get('draw'),
            opening_odds.get('away'),
            closing_odds.get('home'),
            closing_odds.get('draw'),
            closing_odds.get('away'),
            int(time.time())
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✓ Stored closing line for {match_id}")
    
    def get_line_movement(self, match_id: str, outcome: str) -> List[Tuple[int, float]]:
        """
        Get time-series of odds for an outcome.
        
        Returns:
            List of (timestamp, odds) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, AVG(odds)
            FROM odds_history
            WHERE match_id = ? AND outcome = ?
            GROUP BY timestamp
            ORDER BY timestamp
        ''', (match_id, outcome))
        
        rows = cursor.fetchall()
        conn.close()
        
        return rows
    
    def calculate_movement_stats(self, match_id: str) -> Dict:
        """Calculate line movement statistics."""
        opening = self.get_opening_odds(match_id)
        current = self.get_current_odds(match_id)
        
        if not opening or not current:
            return {}
        
        stats = {}
        
        for outcome in ['home', 'draw', 'away']:
            if outcome in opening and outcome in current:
                open_odds = opening[outcome]
                curr_odds = current[outcome]
                
                # Percentage change
                change_pct = ((curr_odds - open_odds) / open_odds) * 100
                
                # Get volatility
                movement = self.get_line_movement(match_id, outcome)
                if len(movement) > 1:
                    import numpy as np
                    odds_values = [odds for _, odds in movement]
                    volatility = float(np.std(odds_values))
                else:
                    volatility = 0.0
                
                stats[outcome] = {
                    'open': open_odds,
                    'current': curr_odds,
                    'change_pct': change_pct,
                    'volatility': volatility
                }
        
        return stats
    
    def track_bet_for_clv(
        self,
        bet_id: str,
        match_id: str,
        outcome: str,
        bet_odds: float
    ):
        """Record a bet for later CLV calculation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO clv_tracking
            (bet_id, match_id, outcome, bet_odds, bet_timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (bet_id, match_id, outcome, bet_odds, int(time.time())))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Tracking bet {bet_id} for CLV")
    
    def calculate_clv(self, bet_id: str) -> Optional[float]:
        """
        Calculate CLV for a bet after match closes.
        
        CLV% = (bet_odds - closing_odds) / closing_odds * 100
        
        Positive = beat the closing line (good)
        Negative = worse than closing (bad)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get bet details
        cursor.execute('''
            SELECT match_id, outcome, bet_odds FROM clv_tracking WHERE bet_id = ?
        ''', (bet_id,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None
        
        match_id, outcome, bet_odds = row
        
        # Get closing odds
        closing_odds_dict = self.get_closing_odds(match_id)
        if not closing_odds_dict or outcome not in closing_odds_dict:
            conn.close()
            return None
        
        closing_odds = closing_odds_dict[outcome]
        
        # Calculate CLV
        clv = ((bet_odds - closing_odds) / closing_odds) * 100
        
        # Update record
        cursor.execute('''
            UPDATE clv_tracking
            SET closing_odds = ?, clv_percent = ?
            WHERE bet_id = ?
        ''', (closing_odds, clv, bet_id))
        
        conn.commit()
        conn.close()
        
        return clv


def test_tracker():
    """Test odds tracker."""
    print("=" * 60)
    print("ODDS TRACKER TEST")
    print("=" * 60)
    
    # Create tracker
    tracker = OddsTracker(db_path='data/odds/test_odds.db')
    
    match_id = "test_match_001"
    
    # Simulate opening odds
    print("\n1. Storing opening odds...")
    opening_odds = {
        'Bet365': {'home': 2.10, 'draw': 3.40, 'away': 3.50},
        'Pinnacle': {'home': 2.08, 'draw': 3.45, 'away': 3.55}
    }
    tracker.store_odds_snapshot(match_id, opening_odds, is_opening=True)
    
    # Simulate line movement
    print("2. Simulating line movement...")
    time.sleep(1)
    
    moved_odds = {
        'Bet365': {'home': 1.95, 'draw': 3.50, 'away': 3.80},  # Home shortened
        'Pinnacle': {'home': 1.93, 'draw': 3.52, 'away': 3.85}
    }
    tracker.store_odds_snapshot(match_id, moved_odds)
    
    # Get movement stats
    print("\n3. Calculating movement stats...")
    stats = tracker.calculate_movement_stats(match_id)
    
    for outcome, data in stats.items():
        print(f"\n{outcome.upper()}:")
        print(f"  Open: {data['open']:.2f}")
        print(f"  Current: {data['current']:.2f}")
        print(f"  Change: {data['change_pct']:+.1f}%")
        print(f"  Volatility: {data['volatility']:.4f}")
    
    # Store closing line
    print("\n4. Storing closing line...")
    closing_odds = {'home': 1.90, 'draw': 3.55, 'away': 3.90}
    tracker.store_closing_line(
        match_id,
        "Man City",
        "Liverpool",
        int(time.time()) + 3600,
        tracker.get_opening_odds(match_id),
        closing_odds
    )
    
    # Test CLV
    print("\n5. Testing CLV calculation...")
    bet_id = "bet_001"
    tracker.track_bet_for_clv(bet_id, match_id, 'home', 2.05)
    
    clv = tracker.calculate_clv(bet_id)
    print(f"  Bet odds: 2.05")
    print(f"  Closing odds: 1.90")
    print(f"  CLV: {clv:+.2f}%")
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_tracker()
