#!/usr/bin/env python3
"""
Snapshot Collection Scheduler (Task 2)

Runs odds snapshot collection on a schedule.
Designed for cron or continuous background execution.
"""

import time
import signal
import logging
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.collect_odds_snapshots import run_collection, get_collection_stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Default interval (30 minutes)
DEFAULT_INTERVAL_MINUTES = 30

# Graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    logger.info("Shutdown requested...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def run_scheduler(interval_minutes: int = DEFAULT_INTERVAL_MINUTES):
    """
    Run collection loop indefinitely.
    
    Args:
        interval_minutes: Minutes between collection runs
    """
    global shutdown_requested
    
    interval_seconds = interval_minutes * 60
    
    logger.info("=" * 50)
    logger.info("🚀 SNAPSHOT SCHEDULER STARTED")
    logger.info(f"   Interval: {interval_minutes} minutes")
    logger.info("   Press Ctrl+C to stop")
    logger.info("=" * 50)
    
    run_count = 0
    
    while not shutdown_requested:
        run_count += 1
        
        logger.info(f"\n--- Run #{run_count} at {datetime.now(timezone.utc).isoformat()} ---")
        
        try:
            result = run_collection()
            logger.info(f"✓ Inserted {result['total_rows_inserted']} rows")
            
        except Exception as e:
            logger.error(f"Collection failed: {e}")
        
        # Sleep in small chunks to allow graceful shutdown
        for _ in range(interval_seconds):
            if shutdown_requested:
                break
            time.sleep(1)
    
    logger.info("\n👋 Scheduler stopped gracefully")
    
    # Print final stats
    stats = get_collection_stats()
    logger.info(f"Total snapshots collected: {stats.get('total_rows', 0):,}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run snapshot scheduler")
    parser.add_argument(
        "--interval", 
        type=int, 
        default=DEFAULT_INTERVAL_MINUTES,
        help=f"Minutes between runs (default: {DEFAULT_INTERVAL_MINUTES})"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (for cron)"
    )
    args = parser.parse_args()
    
    if args.once:
        # Single run mode (for cron)
        result = run_collection()
        print(f"Collected {result['total_rows_inserted']} rows")
    else:
        # Continuous mode
        run_scheduler(args.interval)
