#!/usr/bin/env python3
"""
Stavki V5 Scheduler (Production Resilience)

Runs the value finder pipeline at scheduled intervals (12:00 and 22:00 UTC).
Handles errors gracefully and logs execution status.

Usage:
    python scripts/run_scheduler.py
"""

import time
import subprocess
import schedule
import logging
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

def run_pipeline():
    logging.info("Starting scheduled run...")
    print(f"\n[Scheduler] Triggering run at {datetime.utcnow()} UTC")
    
    try:
        # Run with --now and --global-mode (default in V5 main) and --telegram
        cmd = [
            "python3", "scripts/run_value_finder.py",
            "--now", 
            "--telegram",
            "--global-mode"
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300 # 5 min max
        )
        
        # Log outcome
        if result.returncode == 0:
            logging.info("Run completed successfully.")
            summary_line = [l for l in result.stdout.split('\n') if "Total value bets" in l]
            if summary_line:
                logging.info(f"Summary: {summary_line[0]}")
            else:
                logging.info("Run finished (No summary found)")
            print("[Scheduler] Success.")
        else:
            logging.error(f"Run failed with code {result.returncode}")
            logging.error(f"Stderr: {result.stderr}")
            print(f"[Scheduler] Failed. Check scheduler.log")
            
    except subprocess.TimeoutExpired:
        logging.error("Run timed out (killed after 5m)")
        print("[Scheduler] Timeout.")
        
    except Exception as e:
        logging.error(f"Scheduler exception: {e}")
        print(f"[Scheduler] Error: {e}")

def main():
    print("Stavki V5 Scheduler Started")
    print("Schedule: 12:00 UTC and 22:00 UTC")
    logging.info("Scheduler service started")
    
    schedule.every().day.at("12:00").do(run_pipeline)
    schedule.every().day.at("22:00").do(run_pipeline)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
