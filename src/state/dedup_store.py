"""
Deduplication store using SQLite to track sent alerts.

Prevents spam by tracking which bets have already been alerted.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class DedupStore:
    """SQLite-based deduplication store for value bet alerts."""
    
    def __init__(self, db_path: str = "outputs/state/dedup.db"):
        """
        Initialize dedup store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sent_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                market TEXT NOT NULL,
                outcome TEXT NOT NULL,
                bookmaker TEXT NOT NULL,
                price_bucket REAL NOT NULL,
                sent_at TIMESTAMP NOT NULL,
                ev_pct REAL NOT NULL,
                UNIQUE(event_id, market, outcome, bookmaker, price_bucket)
            )
        """)
        
        # Index for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sent_alerts_lookup
            ON sent_alerts(event_id, market, outcome, bookmaker, price_bucket)
        """)
        
        # Index for cleanup
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sent_alerts_time
            ON sent_alerts(sent_at)
        """)
        
        conn.commit()
        conn.close()
    
    def _price_bucket(self, price: float) -> float:
        """
        Convert price to bucket for deduplication.
        
        Uses 5% buckets to allow for minor price fluctuations.
        
        Args:
            price: Decimal odds
            
        Returns:
            Bucketed price
        """
        # Round to nearest 0.1 for odds < 10, nearest 0.5 for odds >= 10
        if price < 10:
            return round(price * 10) / 10
        else:
            return round(price * 2) / 2
    
    def is_duplicate(
        self,
        event_id: str,
        market: str,
        outcome: str,
        bookmaker: str,
        price: float,
        max_age_hours: int = 48
    ) -> bool:
        """
        Check if bet has already been sent recently.
        
        Args:
            event_id: Event ID
            market: Market key
            outcome: Outcome name
            bookmaker: Bookmaker key
            price: Decimal odds
            max_age_hours: Max age of duplicate (hours)
            
        Returns:
            True if duplicate exists
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        price_bucket = self._price_bucket(price)
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        cursor.execute("""
            SELECT COUNT(*) FROM sent_alerts
            WHERE event_id = ?
              AND market = ?
              AND outcome = ?
              AND bookmaker = ?
              AND price_bucket = ?
              AND sent_at >= ?
        """, (event_id, market, outcome, bookmaker, price_bucket, cutoff))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    def record_sent(
        self,
        event_id: str,
        market: str,
        outcome: str,
        bookmaker: str,
        price: float,
        ev_pct: float
    ):
        """
        Record that alert was sent for this bet.
        
        Args:
            event_id: Event ID
            market: Market key
            outcome: Outcome name
            bookmaker: Bookmaker key
            price: Decimal odds
            ev_pct: EV percentage
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        price_bucket = self._price_bucket(price)
        
        try:
            cursor.execute("""
                INSERT INTO sent_alerts
                (event_id, market, outcome, bookmaker, price_bucket, sent_at, ev_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (event_id, market, outcome, bookmaker, price_bucket, datetime.now(), ev_pct))
            
            conn.commit()
        except sqlite3.IntegrityError:
            # Already exists, update timestamp
            cursor.execute("""
                UPDATE sent_alerts
                SET sent_at = ?, ev_pct = ?
                WHERE event_id = ?
                  AND market = ?
                  AND outcome = ?
                  AND bookmaker = ?
                  AND price_bucket = ?
            """, (datetime.now(), ev_pct, event_id, market, outcome, bookmaker, price_bucket))
            conn.commit()
        
        conn.close()
    
    def filter_new_bets(
        self,
        bets: List[Dict[str, Any]],
        max_age_hours: int = 48
    ) -> List[Dict[str, Any]]:
        """
        Filter out bets that have already been sent.
        
        Args:
            bets: List of bet dictionaries
            max_age_hours: Max age of duplicate (hours)
            
        Returns:
            List of new (unsent) bets
        """
        new_bets = []
        
        for bet in bets:
            if not self.is_duplicate(
                bet['event_id'],
                bet['market'],
                bet['selection'],
                bet['bookmaker_key'],
                bet['odds'],
                max_age_hours
            ):
                new_bets.append(bet)
        
        return new_bets
    
    def cleanup_old(self, days: int = 7):
        """
        Remove entries older than specified days.
        
        Args:
            days: Age threshold in days
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff = datetime.now() - timedelta(days=days)
        
        cursor.execute("""
            DELETE FROM sent_alerts
            WHERE sent_at < ?
        """, (cutoff,))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the dedup store.
        
        Returns:
            Dictionary with stats
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM sent_alerts")
        total = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM sent_alerts
            WHERE sent_at >= ?
        """, (datetime.now() - timedelta(hours=24),))
        last_24h = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM sent_alerts
            WHERE sent_at >= ?
        """, (datetime.now() - timedelta(hours=48),))
        last_48h = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT MIN(sent_at), MAX(sent_at) FROM sent_alerts
        """)
        min_time, max_time = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_entries': total,
            'last_24h': last_24h,
            'last_48h': last_48h,
            'oldest_entry': min_time,
            'newest_entry': max_time,
            'db_path': str(self.db_path),
        }
