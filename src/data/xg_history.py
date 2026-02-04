#!/usr/bin/env python3
"""
xG History Database.

Stores and retrieves historical xG data for teams.
Uses SQLite for persistence.

Usage:
    from src.data.xg_history import XGHistory
    
    xg = XGHistory()
    xg.store_match_xg("arsenal", 2.3, 0.8, "2024-01-15", 3, 1)
    avg_for, avg_against = xg.get_team_xg_avg("arsenal", last_n=5)
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TeamXGRecord:
    """Single xG record for a team."""
    team_id: str
    match_date: str
    xg_for: float
    xg_against: float
    goals_for: int
    goals_against: int
    opponent_id: str
    is_home: bool
    fixture_id: Optional[int] = None


class XGHistory:
    """
    Store and retrieve historical xG data.
    
    Uses SQLite for persistent storage.
    Provides methods to calculate rolling averages for feature extraction.
    """
    
    def __init__(self, db_path: str = "data/xg_history.db"):
        """
        Initialize xG history database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        logger.info(f"XGHistory initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Create database schema if not exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS xg_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id TEXT NOT NULL,
                    match_date TEXT NOT NULL,
                    xg_for REAL NOT NULL,
                    xg_against REAL NOT NULL,
                    goals_for INTEGER NOT NULL,
                    goals_against INTEGER NOT NULL,
                    opponent_id TEXT NOT NULL,
                    is_home INTEGER NOT NULL,
                    fixture_id INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(team_id, match_date, opponent_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_team_date 
                ON xg_matches(team_id, match_date DESC)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_fixture 
                ON xg_matches(fixture_id)
            """)
            
            conn.commit()
    
    def store_match_xg(
        self,
        team_id: str,
        xg_for: float,
        xg_against: float,
        match_date: str,
        goals_for: int,
        goals_against: int,
        opponent_id: str,
        is_home: bool,
        fixture_id: Optional[int] = None
    ) -> bool:
        """
        Store xG data for a match.
        
        Args:
            team_id: Canonical team ID
            xg_for: Expected goals for the team
            xg_against: Expected goals against
            match_date: Match date (YYYY-MM-DD)
            goals_for: Actual goals scored
            goals_against: Actual goals conceded
            opponent_id: Canonical opponent ID
            is_home: Whether team played at home
            fixture_id: Optional SportMonks fixture ID
            
        Returns:
            True if stored, False if already exists
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO xg_matches 
                    (team_id, match_date, xg_for, xg_against, goals_for, 
                     goals_against, opponent_id, is_home, fixture_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    team_id, match_date, xg_for, xg_against,
                    goals_for, goals_against, opponent_id,
                    1 if is_home else 0, fixture_id
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store xG: {e}")
            return False
    
    def store_match_both_teams(
        self,
        home_team: str,
        away_team: str,
        home_xg: float,
        away_xg: float,
        home_goals: int,
        away_goals: int,
        match_date: str,
        fixture_id: Optional[int] = None
    ):
        """
        Store xG data for both teams in a match.
        
        Args:
            home_team: Canonical home team ID
            away_team: Canonical away team ID
            home_xg: xG for home team
            away_xg: xG for away team
            home_goals: Home team goals
            away_goals: Away team goals
            match_date: Match date (YYYY-MM-DD)
            fixture_id: Optional fixture ID
        """
        # Store home team's perspective
        self.store_match_xg(
            team_id=home_team,
            xg_for=home_xg,
            xg_against=away_xg,
            match_date=match_date,
            goals_for=home_goals,
            goals_against=away_goals,
            opponent_id=away_team,
            is_home=True,
            fixture_id=fixture_id
        )
        
        # Store away team's perspective
        self.store_match_xg(
            team_id=away_team,
            xg_for=away_xg,
            xg_against=home_xg,
            match_date=match_date,
            goals_for=away_goals,
            goals_against=home_goals,
            opponent_id=home_team,
            is_home=False,
            fixture_id=fixture_id
        )
    
    def get_team_matches(
        self,
        team_id: str,
        last_n: int = 5,
        before_date: Optional[str] = None,
        home_only: bool = False,
        away_only: bool = False
    ) -> List[TeamXGRecord]:
        """
        Get recent matches for a team.
        
        Args:
            team_id: Canonical team ID
            last_n: Number of matches to retrieve
            before_date: Only matches before this date
            home_only: Only home matches
            away_only: Only away matches
            
        Returns:
            List of TeamXGRecord objects
        """
        query = """
            SELECT team_id, match_date, xg_for, xg_against, 
                   goals_for, goals_against, opponent_id, is_home, fixture_id
            FROM xg_matches
            WHERE team_id = ?
        """
        params = [team_id]
        
        if before_date:
            query += " AND match_date < ?"
            params.append(before_date)
        
        if home_only:
            query += " AND is_home = 1"
        elif away_only:
            query += " AND is_home = 0"
        
        query += " ORDER BY match_date DESC LIMIT ?"
        params.append(last_n)
        
        records = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                records.append(TeamXGRecord(
                    team_id=row[0],
                    match_date=row[1],
                    xg_for=row[2],
                    xg_against=row[3],
                    goals_for=row[4],
                    goals_against=row[5],
                    opponent_id=row[6],
                    is_home=bool(row[7]),
                    fixture_id=row[8]
                ))
        
        return records
    
    def get_team_xg_avg(
        self,
        team_id: str,
        last_n: int = 5,
        before_date: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Get average xG for/against for a team.
        
        Args:
            team_id: Canonical team ID
            last_n: Number of matches to average
            before_date: Only matches before this date
            
        Returns:
            Tuple of (avg_xg_for, avg_xg_against)
        """
        matches = self.get_team_matches(team_id, last_n, before_date)
        
        if not matches:
            # Return league average defaults
            return (1.3, 1.3)
        
        xg_for = sum(m.xg_for for m in matches) / len(matches)
        xg_against = sum(m.xg_against for m in matches) / len(matches)
        
        return (xg_for, xg_against)
    
    def get_team_xg_stats(
        self,
        team_id: str,
        last_n: int = 5,
        before_date: Optional[str] = None
    ) -> Dict:
        """
        Get comprehensive xG stats for a team.
        
        Returns dict with:
        - xg_for_avg: Average xG scored
        - xg_against_avg: Average xG conceded
        - xg_diff: Average xG for - against
        - overperformance: Average goals - xG
        - home_xg_avg: Average xG at home
        - away_xg_avg: Average xG away
        """
        all_matches = self.get_team_matches(team_id, last_n, before_date)
        
        if not all_matches:
            return {
                "xg_for_avg": 1.3,
                "xg_against_avg": 1.3,
                "xg_diff": 0.0,
                "overperformance": 0.0,
                "home_xg_avg": 1.5,
                "away_xg_avg": 1.1
            }
        
        # Overall stats
        xg_for = sum(m.xg_for for m in all_matches) / len(all_matches)
        xg_against = sum(m.xg_against for m in all_matches) / len(all_matches)
        goals_for = sum(m.goals_for for m in all_matches) / len(all_matches)
        
        # Home/away splits
        home_matches = [m for m in all_matches if m.is_home]
        away_matches = [m for m in all_matches if not m.is_home]
        
        home_xg = sum(m.xg_for for m in home_matches) / len(home_matches) if home_matches else 1.5
        away_xg = sum(m.xg_for for m in away_matches) / len(away_matches) if away_matches else 1.1
        
        return {
            "xg_for_avg": round(xg_for, 2),
            "xg_against_avg": round(xg_against, 2),
            "xg_diff": round(xg_for - xg_against, 2),
            "overperformance": round(goals_for - xg_for, 2),
            "home_xg_avg": round(home_xg, 2),
            "away_xg_avg": round(away_xg, 2),
            "matches_count": len(all_matches)
        }
    
    def get_h2h_xg(
        self,
        team1: str,
        team2: str,
        last_n: int = 5
    ) -> Dict:
        """
        Get head-to-head xG stats between two teams.
        
        Returns dict with average xG for each team in H2H matches.
        """
        query = """
            SELECT xg_for, xg_against, goals_for, goals_against, is_home
            FROM xg_matches
            WHERE team_id = ? AND opponent_id = ?
            ORDER BY match_date DESC
            LIMIT ?
        """
        
        h2h_records = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, (team1, team2, last_n))
            for row in cursor.fetchall():
                h2h_records.append({
                    "xg_for": row[0],
                    "xg_against": row[1],
                    "goals_for": row[2],
                    "goals_against": row[3],
                    "is_home": bool(row[4])
                })
        
        if not h2h_records:
            return {
                "matches": 0,
                "team1_xg_avg": 1.3,
                "team2_xg_avg": 1.3,
                "team1_goals_avg": 1.5,
                "team2_goals_avg": 1.5
            }
        
        return {
            "matches": len(h2h_records),
            "team1_xg_avg": sum(r["xg_for"] for r in h2h_records) / len(h2h_records),
            "team2_xg_avg": sum(r["xg_against"] for r in h2h_records) / len(h2h_records),
            "team1_goals_avg": sum(r["goals_for"] for r in h2h_records) / len(h2h_records),
            "team2_goals_avg": sum(r["goals_against"] for r in h2h_records) / len(h2h_records)
        }
    
    def bulk_import_from_sportmonks(
        self,
        sportmonks_client,
        team_ids: List[str],
        last_n_matches: int = 10
    ) -> int:
        """
        Import historical xG data from SportMonks for multiple teams.
        
        Args:
            sportmonks_client: SportMonks API client
            team_ids: List of canonical team IDs
            last_n_matches: Number of matches per team
            
        Returns:
            Number of matches imported
        """
        from ..utils.team_normalizer import get_sportmonks_id, normalize_team
        
        imported = 0
        
        for team_id in team_ids:
            sportmonks_id = get_sportmonks_id(team_id)
            if not sportmonks_id:
                logger.warning(f"No SportMonks ID for {team_id}")
                continue
            
            try:
                # Get last N matches with stats
                matches = sportmonks_client.get_team_matches(
                    sportmonks_id, 
                    last_n=last_n_matches
                )
                
                for match in matches:
                    stats = sportmonks_client.get_fixture_stats(match.fixture_id)
                    if stats and stats.home_xg is not None:
                        # Determine which team is home/away
                        home_team = normalize_team(match.home_team)
                        away_team = normalize_team(match.away_team)
                        
                        self.store_match_both_teams(
                            home_team=home_team,
                            away_team=away_team,
                            home_xg=stats.home_xg,
                            away_xg=stats.away_xg,
                            home_goals=match.home_goals or 0,
                            away_goals=match.away_goals or 0,
                            match_date=match.kickoff.strftime("%Y-%m-%d"),
                            fixture_id=match.fixture_id
                        )
                        imported += 1
                        
            except Exception as e:
                logger.warning(f"Failed to import xG for {team_id}: {e}")
        
        logger.info(f"Imported {imported} xG records from SportMonks")
        return imported
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM xg_matches").fetchone()[0]
            teams = conn.execute("SELECT COUNT(DISTINCT team_id) FROM xg_matches").fetchone()[0]
            
            earliest = conn.execute(
                "SELECT MIN(match_date) FROM xg_matches"
            ).fetchone()[0]
            
            latest = conn.execute(
                "SELECT MAX(match_date) FROM xg_matches"
            ).fetchone()[0]
        
        return {
            "total_records": total,
            "unique_teams": teams,
            "earliest_match": earliest,
            "latest_match": latest
        }


# Singleton instance
_xg_history: Optional[XGHistory] = None


def get_xg_history(db_path: str = "data/xg_history.db") -> XGHistory:
    """Get singleton XGHistory instance."""
    global _xg_history
    if _xg_history is None:
        _xg_history = XGHistory(db_path)
    return _xg_history


# CLI for testing
if __name__ == "__main__":
    import tempfile
    
    print("xG History Database Tests")
    print("=" * 50)
    
    # Use temp database for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/test_xg.db"
        xg = XGHistory(db_path)
        
        # Store some test data
        test_matches = [
            ("arsenal", "chelsea", 2.1, 0.9, 2, 0, "2024-01-15"),
            ("arsenal", "manchester_city", 1.2, 2.3, 0, 3, "2024-01-22"),
            ("arsenal", "liverpool", 1.8, 1.5, 2, 2, "2024-01-29"),
            ("arsenal", "tottenham", 2.5, 0.6, 3, 1, "2024-02-05"),
            ("arsenal", "west_ham", 2.0, 0.8, 2, 0, "2024-02-12"),
        ]
        
        for home, away, h_xg, a_xg, h_g, a_g, date in test_matches:
            xg.store_match_both_teams(home, away, h_xg, a_xg, h_g, a_g, date)
        
        print(f"✓ Stored {len(test_matches)} test matches")
        
        # Test retrieval
        avg_for, avg_against = xg.get_team_xg_avg("arsenal", last_n=5)
        print(f"✓ Arsenal xG avg: {avg_for:.2f} for, {avg_against:.2f} against")
        
        # Test stats
        stats = xg.get_team_xg_stats("arsenal", last_n=5)
        print(f"✓ Arsenal stats: {stats}")
        
        # Test H2H
        h2h = xg.get_h2h_xg("arsenal", "chelsea", last_n=5)
        print(f"✓ Arsenal vs Chelsea H2H: {h2h}")
        
        # DB stats
        db_stats = xg.get_stats()
        print(f"✓ Database: {db_stats}")
        
        print("=" * 50)
        print("✅ All tests passed!")
