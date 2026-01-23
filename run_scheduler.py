#!/usr/bin/env python3
"""
STAVKI Automation Scheduler

Runs odds fetching and value finding in a loop with deduplication and batched Telegram alerts.

Usage:
    python run_scheduler.py --interval 60 --telegram
    
    python run_scheduler.py --help
"""

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config.env import load_env_config
from src.state.dedup_store import DedupStore
from src.integration.telegram_notify import send_value_alert, is_telegram_configured


def setup_logging(output_dir: str = "outputs/scheduler"):
    """
    Setup logging for scheduler.
    
    Args:
        output_dir: Directory for log files
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = Path(output_dir) / f"scheduler_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def run_odds_pipeline(sport: str, logger: logging.Logger) -> bool:
    """
    Run odds pipeline.
    
    Args:
        sport: Sport key
        logger: Logger instance
        
    Returns:
        True if successful
    """
    logger.info(f"Running odds pipeline for {sport}...")
    
    try:
        result = subprocess.run(
            ["python", "run_odds_pipeline.py", "--sport", sport],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes max
        )
        
        if result.returncode == 0:
            logger.info(f"✓ Odds pipeline completed successfully")
            return True
        else:
            logger.error(f"✗ Odds pipeline failed: {result.stderr[:200]}")
            return False
    except Exception as e:
        logger.error(f"✗ Odds pipeline error: {e}")
        return False


def run_value_finder(
    sport: str,
    ev_threshold: float,
    confirm_high_odds: bool,
    outlier_drop: bool,
    logger: logging.Logger
) -> tuple[bool, list]:
    """
    Run value finder and return candidates.
    
    Args:
        sport: Sport key
        ev_threshold: EV threshold
        confirm_high_odds: Enable high odds confirmation
        outlier_drop: Enable outlier detection
        logger: Logger instance
        
    Returns:
        (success, candidates)
    """
    import json
    
    logger.info(f"Running value finder...")
    
    # Build command
    cmd = [
        "python", "run_value_finder.py",
        "--sport", sport,
        "--ev-threshold", str(ev_threshold),
        "--top-n", "10",
    ]
    
    if confirm_high_odds:
        cmd.append("--confirm-high-odds")
    
    if outlier_drop:
        cmd.append("--outlier-drop")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes max
        )
        
        if result.returncode != 0:
            logger.error(f"✗ Value finder failed: {result.stderr[:200]}")
            return False, []
        
        # Load latest JSON output
        value_files = sorted(Path("outputs/value").glob(f"value_{sport}_*.json"))
        if not value_files:
            logger.warning(f"⚠ No value files found")
            return True, []
        
        latest_file = value_files[-1]
        
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        bets = data.get('bets', [])
        logger.info(f"✓ Value finder found {len(bets)} candidates")
        
        return True, bets
        
    except Exception as e:
        logger.error(f"✗ Value finder error: {e}")
        return False, []


def main():
    """Main scheduler loop."""
    parser = argparse.ArgumentParser(
        description="Automated scheduler for odds fetching and value finding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run every hour with guardrails and Telegram
  python run_scheduler.py --interval 60 --telegram \\
    --confirm-high-odds --outlier-drop
  
  # Short interval for testing (2 cycles)
  python run_scheduler.py --interval 5 --max-runs 2
        """
    )
    
    parser.add_argument(
        '--sport',
        default='soccer_epl',
        help='Sport key (default: soccer_epl)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Minutes between runs (default: 60)'
    )
    parser.add_argument(
        '--ev-threshold',
        type=float,
        default=0.08,
        help='EV threshold (default: 0.08)'
    )
    parser.add_argument(
        '--max-runs',
        type=int,
        help='Maximum number of runs (for testing)'
    )
    
    # Guardrails
    parser.add_argument(
        '--confirm-high-odds',
        action='store_true',
        help='Enable high-odds confirmation (recommended)'
    )
    parser.add_argument(
        '--outlier-drop',
        action='store_true',
        help='Enable outlier detection (recommended)'
    )
    
    # Dedup
    parser.add_argument(
        '--dedup-db',
        default='outputs/state/dedup.db',
        help='Dedup database path'
    )
    parser.add_argument(
        '--dedup-max-age',
        type=int,
        default=48,
        help='Max age for dedup check (hours, default: 48)'
    )
    
    # Telegram
    parser.add_argument(
        '--telegram',
        action='store_true',
        help='Send Telegram alerts'
    )
    parser.add_argument(
        '--telegram-top-n',
        type=int,
        default=5,
        help='Number of bets to include in Telegram alert (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("STAVKI AUTOMATION SCHEDULER STARTED")
    logger.info("=" * 70)
    logger.info(f"Sport: {args.sport}")
    logger.info(f"Interval: {args.interval} minutes")
    logger.info(f"EV Threshold: {args.ev_threshold:.2%}")
    logger.info(f"Guardrails: confirm_high_odds={args.confirm_high_odds}, outlier_drop={args.outlier_drop}")
    logger.info(f"Telegram: {args.telegram}")
    if args.max_runs:
        logger.info(f"Max runs: {args.max_runs}")
    logger.info("=" * 70)
    
    # Load env config
    try:
        env_config = load_env_config()
    except SystemExit:
        env_config = {}
    
    # Initialize dedup store
    dedup = DedupStore(args.dedup_db)
    logger.info(f"✓ Dedup store initialized: {args.dedup_db}")
    
    # Check Telegram
    if args.telegram and not is_telegram_configured():
        logger.warning("⚠ Telegram requested but not configured (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")
        logger.warning("   Continuing without Telegram alerts")
        args.telegram = False
    
    run_count = 0
    
    try:
        while True:
            run_count += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"RUN #{run_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*70}")
            
            # Step 1: Fetch odds
            odds_success = run_odds_pipeline(args.sport, logger)
            
            if not odds_success:
                logger.warning("⚠ Skipping value finder due to odds pipeline failure")
            else:
                # Step 2: Find value bets
                value_success, candidates = run_value_finder(
                    args.sport,
                    args.ev_threshold,
                    args.confirm_high_odds,
                    args.outlier_drop,
                    logger
                )
                
                if value_success and candidates:
                    # Step 3: Filter with dedup
                    logger.info(f"🔍 Deduplication check...")
                    new_bets = dedup.filter_new_bets(candidates, args.dedup_max_age)
                    logger.info(f"   Candidates: {len(candidates)}, New: {len(new_bets)}, Filtered: {len(candidates) - len(new_bets)}")
                    
                    if new_bets:
                        # Step 4: Send Telegram alert (batched)
                        if args.telegram:
                            logger.info(f"📱 Sending Telegram alert for {min(len(new_bets), args.telegram_top_n)} bets...")
                            success = send_value_alert(new_bets, top_n=args.telegram_top_n)
                            
                            if success:
                                logger.info(f"   ✓ Telegram alert sent")
                                
                                # Record in dedup store
                                for bet in new_bets[:args.telegram_top_n]:
                                    dedup.record_sent(
                                        bet['event_id'],
                                        bet['market'],
                                        bet['selection'],
                                        bet['bookmaker_key'],
                                        bet['odds'],
                                        bet['ev_pct']
                                    )
                                logger.info(f"   ✓ Recorded {min(len(new_bets), args.telegram_top_n)} bets in dedup store")
                            else:
                                logger.error(f"   ✗ Telegram alert failed")
                        else:
                            logger.info(f"💡 {len(new_bets)} new bets found (Telegram disabled)")
                    else:
                        logger.info(f"   No new bets (all duplicates)")
            
            # Cleanup old dedup entries
            if run_count % 10 == 0:  # Every 10 runs
                logger.info(f"🧹 Cleaning up old dedup entries...")
                deleted = dedup.cleanup_old(days=7)
                logger.info(f"   Deleted {deleted} entries older than 7 days")
            
            # Check if we should stop
            if args.max_runs and run_count >= args.max_runs:
                logger.info(f"\n{'='*70}")
                logger.info(f"Reached max runs ({args.max_runs}), stopping")
                logger.info(f"{'='*70}")
                break
            
            # Sleep until next run
            sleep_seconds = args.interval * 60
            logger.info(f"\n💤 Sleeping for {args.interval} minutes...")
            logger.info(f"   Next run at: {(datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}")
            
            time.sleep(sleep_seconds)
    
    except KeyboardInterrupt:
        logger.info(f"\n{'='*70}")
        logger.info("Scheduler stopped by user (Ctrl+C)")
        logger.info(f"{'='*70}")
    except Exception as e:
        logger.error(f"\n{'='*70}")
        logger.error(f"Scheduler crashed: {e}")
        logger.error(f"{'='*70}")
        raise
    finally:
        # Final stats
        stats = dedup.get_stats()
        logger.info(f"\n{'='*70}")
        logger.info("DEDUP STORE STATS")
        logger.info(f"{'='*70}")
        logger.info(f"Total entries: {stats['total_entries']}")
        logger.info(f"Last 24h: {stats['last_24h']}")
        logger.info(f"Last 48h: {stats['last_48h']}")
        logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
