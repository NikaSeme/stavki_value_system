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
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.integration.telegram_notify import send_custom_message
from src.data.odds_tracker import OddsTracker
import json
from datetime import datetime


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
        print("💡 TIP: Use /stop in Telegram or 'pkill -f run_scheduler.py' if this is a ghost process.")
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

def run_orchestrator(telegram=False, bankroll=None, ev_threshold=None, leagues=None, ev_max=None):
    """Run the full Odds -> Value pipeline."""
    utc_now = datetime.now(timezone.utc)
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
        # Extract last few lines of error
        err_msg = out_odds.strip()[-300:] if out_odds else "Unknown error"
        msg = f"❌ *Bot Run Failed*: Odds Pipeline Error.\nLogs:\n`{err_msg}`"
        
        logging.error(msg)
        print(f"[Scheduler] {msg}")
        if telegram:
            send_custom_message(msg)
        # EXIT with error code so systemd/user knows it failed and lock is released
        sys.exit(1)

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
    if ev_max:
        vf_cmd.extend(["--ev-max", str(ev_max)])
    if leagues:
        vf_cmd.extend(["--leagues", leagues])
        
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

    # Update state file knowing next run is scheduled
    # Access global main_job if possible, or simpler: just write timestamp
    # But job.next_run is internal to schedule. 
    # Valid Hack: We can just write "now + interval" approx, or better, pass job to this func?
    # We will pass main_job to this function in future refactor, 
    # but for now let's write to file OUTSIDE this function in the main loop or use global hook.
    
def save_state(next_run_dt):
    """Save next run time to a JSON file for the bot to read."""
    try:
        data = {
            "next_run": next_run_dt.isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        with open("data/scheduler_state.json", "w") as f:
            json.dump(data, f)
    except Exception as e:
        logging.error(f"Failed to save scheduler state: {e}")


def run_mini_update(telegram_enabled):
    """
    Run a lightweight update:
    1. Fetch fresh odds (track lines)
    2. Check for sharp movements (Steam Moves)
    3. Alert if found
    """
    logging.info("=== Mini Update Start (Line Check) ===")
    print("\n[Scheduler] Running 10-min Line Check...")
    
    # 1. Fast Odds Update
    # We use the same pipeline but rely on its efficiency
    success, _ = run_command(
        [sys.executable, "scripts/run_odds_pipeline.py", "--track-lines"],
        "Mini Odds Fetch"
    )
    
    if not success:
        logging.warning("Mini update odds fetch failed")
        return

    # 2. Check for Sharp Moves
    try:
        tracker = OddsTracker() # Defaults to standard DB path
        # We need to identify WHICH matches to check. 
        # For efficiency, we could check active matches from config or just check recent updates.
        # Since we just ran pipeline, we can check all active matches in DB or list from odds pipeline output.
        # A simpler approach: Check matches where 'last_update' is recent.
        
        # We will iterate through a specific set of IDs if possible, but here we'll 
        # check ongoing events. For now, accessing DB directly is best.
        
        conn = sqlite3.connect(tracker.db_path)
        cursor = conn.cursor()
        # Get matches updated in last 20 mins
        since = int(time.time()) - 1200
        cursor.execute('SELECT DISTINCT match_id FROM odds_history WHERE timestamp > ?', (since,))
        match_ids = [r[0] for r in cursor.fetchall()]
        conn.close()
        
        if not match_ids:
            logging.info("No matches updated recently.")
            return

        logging.info(f"Checking {len(match_ids)} matches for sharp moves...")
        alerts = []
        
        for mid in match_ids:
            signals = tracker.get_movement_signals(mid)
            for outcome, data in signals.get('outcomes', {}).items():
                sig_type = data.get('signal')
                if sig_type in ['SHARP_MONEY', 'STRONG_VALUE']:
                    explanation = data.get('explanation', '')
                    avg_change = data.get('steam', {}).get('avg_change_pct', 0)
                    
                    # Deduplicate: Check if we alerted this recently? 
                    # For V1, we just alert.
                    msg = f"📉 *Sharp Move Detected!* {sig_type}\nMatch: `{mid}`\nOutcome: {outcome}\nChange: {avg_change:.1f}%\nReason: {explanation}"
                    alerts.append(msg)
        
        if alerts and telegram_enabled:
            # Combine into one message to avoid spam
            full_msg = "\n\n".join(alerts[:5]) # Top 5 only
            if len(alerts) > 5:
                full_msg += f"\n\n...and {len(alerts)-5} more."
                
            send_custom_message(f"🚨 **Line Movement Alerts**\n\n{full_msg}")
            logging.info(f"Sent {len(alerts)} sharp move alerts.")
            
    except Exception as e:
        logging.error(f"Error in Line Check: {e}")

    logging.info("=== Mini Update End ===")

def main():
    parser = argparse.ArgumentParser(description="Stavki Scheduler Service")
    parser.add_argument('--interval', type=int, help='Run every N minutes (loops forever)')
    parser.add_argument('--telegram', action='store_true', help='Enable Telegram alerts for these runs')
    parser.add_argument('--now', action='store_true', help='Run immediately on start')
    parser.add_argument('--bankroll', type=float, help='Bankroll for value finder')
    parser.add_argument('--ev-threshold', type=float, help='EV threshold for value finder')
    parser.add_argument('--ev-max', type=float, help='Maximum EV for value finder')
    parser.add_argument('--leagues', type=str, help='Comma-separated list of league keys to include')
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
            run_orchestrator(
                telegram=args.telegram, 
                bankroll=args.bankroll, 
                ev_threshold=args.ev_threshold,
                ev_max=args.ev_max,
                leagues=args.leagues
            )

        # Schedule
        main_job = None
        
        if args.interval:
            print(f"Schedule: Running every {args.interval} minutes.")
            main_job = schedule.every(args.interval).minutes.do(
                run_orchestrator, 
                telegram=args.telegram, 
                bankroll=args.bankroll, 
                ev_threshold=args.ev_threshold,
                ev_max=args.ev_max,
                leagues=args.leagues
            )
            
            # Add Mini Update (Line Check) every 10 minutes
            # DISABLED to save API tokens (User Request)
            # schedule.every(10).minutes.do(
            #    run_mini_update,
            #    telegram_enabled=args.telegram
            # )
            
            # Initial Save
            if main_job and main_job.next_run:
                save_state(main_job.next_run)
                logging.info(f"Next prediction run: {main_job.next_run}")
            
        else:
            # Default fixed schedule (Production)
            print("Schedule: Fixed times (12:00, 22:00 UTC)")
            # Note: Countdown doesn't work well with fixed times without more logic, 
            # but user specifically asked for interval behavior.
            schedule.every().day.at("12:00").do(
                run_orchestrator, 
                telegram=args.telegram, 
                bankroll=args.bankroll, 
                ev_threshold=args.ev_threshold,
                ev_max=args.ev_max,
                leagues=args.leagues
            )
            schedule.every().day.at("22:00").do(
                run_orchestrator, 
                telegram=args.telegram, 
                bankroll=args.bankroll, 
                ev_threshold=args.ev_threshold,
                ev_max=args.ev_max,
                leagues=args.leagues
            )

        # Main loop
        while True:
            schedule.run_pending()
            
             # Simple state persistence check
            if main_job and main_job.next_run:
                save_state(main_job.next_run)
                
            time.sleep(10) # check every 10s
            
    finally:
        release_lock(lock_f)

if __name__ == "__main__":
    main()
