#!/usr/bin/env python3
"""
Stavki V5 Scheduler (Production Resilience)

Runs the full pipeline (Odds -> Value) at scheduled intervals.
Can run in 'loop' mode (interval) or 'daily' mode (fixed times).

Usage:
    python scripts/run_scheduler.py --interval 60 --telegram
    python scripts/run_scheduler.py --now
"""

import time
import subprocess
import schedule
import logging
import argparse
import sys
import os
import signal
import fcntl
from datetime import datetime
from pathlib import Path

# Setup structured logging
log_dir = Path("audit_pack/RUN_LOGS")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=log_dir / "scheduler.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

LOCK_FILE = "/tmp/stavki_scheduler.lock"

def acquire_lock():
    """Acquire a file lock to prevent multiple scheduler instances."""
    try:
        f = open(LOCK_FILE, 'w')
        # Try to acquire an exclusive lock without blocking
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Write PID to the file for debugging
        f.write(str(os.getpid()))
        f.flush()
        return f
    except (IOError, OSError):
        print("❌ Error: Another instance of the scheduler is already running.")
        sys.exit(1)

def release_lock(lock_file):
    """Release the file lock and clean up."""
    if lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)
        except Exception as e:
            logging.error(f"Error releasing lock: {e}")

def signal_handler(sig, frame):
    """Handle termination signals."""
    logging.info(f"Received signal {sig}. Shutting down...")
    print(f"\n[Scheduler] Signal received. Cleaning up...")
    # The lock_file variable is not in scope here if we just use main
    # But for a script, simply exiting usually suffices if OS cleans up FLOCK.
    # However, we'll try to be clean.
    sys.exit(0)

def run_command(cmd_list, step_name):
    """Run a subprocess command and log output."""
    logging.info(f"[{step_name}] Starting: {' '.join(cmd_list)}")
    print(f"[{step_name}] Running...")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600  # 10 min max per step
        )
        duration = time.time() - start_time
        
        if result.returncode == 0:
            logging.info(f"[{step_name}] Success ({duration:.2f}s)")
            return True, result.stdout
        else:
            logging.error(f"[{step_name}] Failed ({duration:.2f}s) Code: {result.returncode}")
            logging.error(f"Stderr: {result.stderr}")
            print(f"[{step_name}] ❌ Failed. Check logs.")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        logging.error(f"[{step_name}] Timeout after {time.time() - start_time:.2f}s")
        print(f"[{step_name}] ⏳ Timed out.")
        return False, "Timeout"
    except Exception as e:
        logging.error(f"[{step_name}] Exception: {e}")
        print(f"[{step_name}] 💥 Error: {e}")
        return False, str(e)

def run_orchestrator(telegram=False, bankroll=None, ev_threshold=None):
    """Run the full Odds -> Value pipeline."""
    utc_now = datetime.utcnow()
    print(f"\n[Scheduler] Triggering Run at {utc_now} UTC")
    logging.info(f"=== Orchestration Start: {utc_now} ===")

    # Step 1: Odds Ingestion
    # We run odds pipeline to fetch ALL sports configured in config/leagues.yaml
    # We enable --track-lines to build time-series history
    success_odds, out_odds = run_command(
        [sys.executable, "scripts/run_odds_pipeline.py", "--track-lines"],
        "Odds Pipeline"
    )
    
    if not success_odds:
        logging.error("Aborting run due to Odds Pipeline failure.")
        print("[Scheduler] Odds Step Failed. Aborting.")
        return

    # Step 2: Value Finder
    # We pass --telegram if enabled
    vf_cmd = [
        sys.executable, "scripts/run_value_finder.py",
        "--now",
        "--global-mode",
        "--auto" # M21: Ensure non-interactive mode
    ]
    if telegram:
        vf_cmd.append("--telegram")
    if bankroll:
        vf_cmd.extend(["--bankroll", str(bankroll)])
    if ev_threshold:
        vf_cmd.extend(["--ev-threshold", str(ev_threshold)])
        
    success_vf, out_vf = run_command(vf_cmd, "Value Finder")
    
    if success_vf:
        # Extract summary from stdout if possible
        summary_line = [l for l in out_vf.split('\n') if "Final Selection:" in l]
        if summary_line:
            print(f"  -> {summary_line[0].strip()}")
            logging.info(f"Summary: {summary_line[0].strip()}")
        else:
            print("  -> Run Complete (Check logs for details)")
            
    logging.info("=== Orchestration End ===")
    print("[Scheduler] Cycle Complete.\n")

def main():
    parser = argparse.ArgumentParser(description="Stavki Scheduler Service")
    parser.add_argument('--interval', type=int, help='Run every N minutes (loops forever)')
    parser.add_argument('--telegram', action='store_true', help='Enable Telegram alerts for these runs')
    parser.add_argument('--now', action='store_true', help='Run immediately on start')
    parser.add_argument('--bankroll', type=float, help='Bankroll for value finder')
    parser.add_argument('--ev-threshold', type=float, help='EV threshold for value finder')
    args = parser.parse_args()

    print("Stavki V5 Scheduler Started")
    logging.info("Scheduler service started")

    # Global lock
    lock_f = acquire_lock()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Immediate run?
        if args.now:
            run_orchestrator(telegram=args.telegram, bankroll=args.bankroll, ev_threshold=args.ev_threshold)

        # Schedule
        if args.interval:
            print(f"Schedule: Running every {args.interval} minutes.")
            schedule.every(args.interval).minutes.do(
                run_orchestrator, 
                telegram=args.telegram, 
                bankroll=args.bankroll, 
                ev_threshold=args.ev_threshold
            )
        else:
            # Default fixed schedule (Production)
            print("Schedule: Fixed times (12:00, 22:00 UTC)")
            schedule.every().day.at("12:00").do(
                run_orchestrator, 
                telegram=args.telegram, 
                bankroll=args.bankroll, 
                ev_threshold=args.ev_threshold
            )
            schedule.every().day.at("22:00").do(
                run_orchestrator, 
                telegram=args.telegram, 
                bankroll=args.bankroll, 
                ev_threshold=args.ev_threshold
            )

        while True:
            schedule.run_pending()
            time.sleep(10) # lighter sleep
    finally:
        release_lock(lock_f)

if __name__ == "__main__":
    main()
