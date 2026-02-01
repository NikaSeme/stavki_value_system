#!/usr/bin/env python3
"""
Odds Snapshot Collector (Task 2)

Collects timestamped odds snapshots from The Odds API and stores in SQLite.
Run periodically (every 30 min) to build time-series odds history.
"""

import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.odds_api_client import fetch_odds, load_config_from_env
from src.config.env import load_env_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "odds_snapshots.sqlite"

# Sports to track
SPORTS_TO_TRACK = [
    "soccer_epl",
    "soccer_germany_bundesliga", 
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_efl_champ",
]


def init_database(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize SQLite database with schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create main table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS odds_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            sport_key TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            commence_time TEXT NOT NULL,
            collected_at TEXT NOT NULL,
            bookmaker TEXT NOT NULL,
            market TEXT DEFAULT 'h2h',
            outcome_name TEXT NOT NULL,
            outcome_price REAL NOT NULL,
            
            UNIQUE(event_id, collected_at, bookmaker, outcome_name)
        )
    """)
    
    # Create indexes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_event_time 
        ON odds_snapshots(event_id, collected_at)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_commence 
        ON odds_snapshots(commence_time)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_sport 
        ON odds_snapshots(sport_key)
    """)
    
    # Create metadata table for tracking collection runs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS collection_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            sports_collected TEXT,
            events_collected INTEGER DEFAULT 0,
            rows_inserted INTEGER DEFAULT 0,
            status TEXT DEFAULT 'running'
        )
    """)
    
    conn.commit()
    return conn


def collect_odds_for_sport(
    sport_key: str,
    conn: sqlite3.Connection,
    collected_at: str
) -> int:
    """
    Fetch and store odds for a single sport.
    
    Returns number of rows inserted.
    """
    try:
        events = fetch_odds(
            sport_key=sport_key,
            regions="eu,uk",
            markets="h2h",
            odds_format="decimal"
        )
    except Exception as e:
        logger.error(f"Failed to fetch odds for {sport_key}: {e}")
        return 0
    
    if not events:
        logger.info(f"No events found for {sport_key}")
        return 0
    
    cursor = conn.cursor()
    rows_inserted = 0
    
    for event in events:
        event_id = event.get("id")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        commence_time = event.get("commence_time")
        
        # Skip if already started
        try:
            commence_dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
            if commence_dt < datetime.now(timezone.utc):
                continue
        except:
            pass
        
        for bookmaker in event.get("bookmakers", []):
            bookmaker_key = bookmaker.get("key")
            
            for market in bookmaker.get("markets", []):
                market_key = market.get("key")
                
                for outcome in market.get("outcomes", []):
                    outcome_name = outcome.get("name")
                    outcome_price = outcome.get("price")
                    
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO odds_snapshots 
                            (event_id, sport_key, home_team, away_team, 
                             commence_time, collected_at, bookmaker, 
                             market, outcome_name, outcome_price)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            event_id, sport_key, home_team, away_team,
                            commence_time, collected_at, bookmaker_key,
                            market_key, outcome_name, outcome_price
                        ))
                        
                        if cursor.rowcount > 0:
                            rows_inserted += 1
                            
                    except sqlite3.Error as e:
                        logger.warning(f"Insert error: {e}")
    
    conn.commit()
    return rows_inserted


def run_collection(sports: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Run a full collection cycle for all configured sports.
    
    Returns summary dict.
    """
    if sports is None:
        sports = SPORTS_TO_TRACK
    
    # Load API config
    load_env_config()
    
    # Initialize DB
    conn = init_database()
    cursor = conn.cursor()
    
    # Record run start
    started_at = datetime.now(timezone.utc).isoformat()
    cursor.execute("""
        INSERT INTO collection_runs (started_at, sports_collected, status)
        VALUES (?, ?, 'running')
    """, (started_at, ",".join(sports)))
    run_id = cursor.lastrowid
    conn.commit()
    
    # Collect
    total_rows = 0
    events_per_sport = {}
    
    for sport in sports:
        logger.info(f"Collecting {sport}...")
        rows = collect_odds_for_sport(sport, conn, started_at)
        events_per_sport[sport] = rows
        total_rows += rows
        logger.info(f"  → {rows} rows inserted")
    
    # Update run record
    completed_at = datetime.now(timezone.utc).isoformat()
    cursor.execute("""
        UPDATE collection_runs 
        SET completed_at = ?, rows_inserted = ?, status = 'completed'
        WHERE id = ?
    """, (completed_at, total_rows, run_id))
    conn.commit()
    
    # Summary
    summary = {
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "total_rows_inserted": total_rows,
        "per_sport": events_per_sport,
    }
    
    conn.close()
    return summary


def get_collection_stats() -> Dict[str, Any]:
    """Get statistics about collected snapshots."""
    if not DB_PATH.exists():
        return {"error": "Database not found"}
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Total rows
    cursor.execute("SELECT COUNT(*) FROM odds_snapshots")
    total_rows = cursor.fetchone()[0]
    
    # Unique events
    cursor.execute("SELECT COUNT(DISTINCT event_id) FROM odds_snapshots")
    unique_events = cursor.fetchone()[0]
    
    # Date range
    cursor.execute("""
        SELECT MIN(collected_at), MAX(collected_at) 
        FROM odds_snapshots
    """)
    date_range = cursor.fetchone()
    
    # Per sport
    cursor.execute("""
        SELECT sport_key, COUNT(*) as cnt 
        FROM odds_snapshots 
        GROUP BY sport_key
    """)
    per_sport = dict(cursor.fetchall())
    
    # Collection runs
    cursor.execute("SELECT COUNT(*) FROM collection_runs")
    total_runs = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "total_rows": total_rows,
        "unique_events": unique_events,
        "date_range": date_range,
        "per_sport": per_sport,
        "total_runs": total_runs,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect odds snapshots")
    parser.add_argument("--stats", action="store_true", help="Show collection stats")
    parser.add_argument("--sports", type=str, help="Comma-separated sport keys")
    args = parser.parse_args()
    
    if args.stats:
        stats = get_collection_stats()
        print("\n📊 SNAPSHOT COLLECTION STATS")
        print("=" * 40)
        print(f"Total rows: {stats.get('total_rows', 0):,}")
        print(f"Unique events: {stats.get('unique_events', 0):,}")
        print(f"Date range: {stats.get('date_range', ('N/A', 'N/A'))}")
        print(f"Collection runs: {stats.get('total_runs', 0)}")
        print("\nPer sport:")
        for sport, count in stats.get('per_sport', {}).items():
            print(f"  {sport}: {count:,}")
    else:
        sports = args.sports.split(",") if args.sports else None
        
        print("\n🔄 STARTING ODDS SNAPSHOT COLLECTION")
        print("=" * 40)
        
        result = run_collection(sports)
        
        print("\n✅ COLLECTION COMPLETE")
        print(f"Run ID: {result['run_id']}")
        print(f"Total inserted: {result['total_rows_inserted']}")
        print("\nPer sport:")
        for sport, count in result['per_sport'].items():
            print(f"  {sport}: {count}")
