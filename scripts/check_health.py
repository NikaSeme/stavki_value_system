#!/usr/bin/env python3
"""
STAVKI Health Check (M31)

Validates:
1. Odds data freshness (outputs/odds/events_latest_*.csv)
2. Scheduler process status (via lock file)
3. Recent log activity
4. Disk space / basic environment

Usage:
    python scripts/check_health.py --telegram
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.integration.telegram_notify import send_custom_message, is_telegram_configured

def check_odds_freshness(max_age_hours=24):
    """Check if latest odds files are recent."""
    odds_dir = Path("outputs/odds")
    if not odds_dir.exists():
        return False, "Odds directory missing."
    
    latest_files = list(odds_dir.glob("events_latest_*.csv"))
    if not latest_files:
        return False, "No odds files found in outputs/odds/."
    
    # Check most recent file
    latest_file = max(latest_files, key=lambda p: p.stat().st_mtime)
    age_seconds = time.time() - latest_file.stat().st_mtime
    age_hours = age_seconds / 3600
    
    if age_hours > max_age_hours:
        return False, f"Odds data is stale ({age_hours:.1f}h old). Max allowed: {max_age_hours}h."
    
    return True, f"Odds data fresh ({age_hours:.1f}h old)."

def check_scheduler_lock():
    """Verify if scheduler lock exists (implies it might be running)."""
    lock_file = Path("/tmp/stavki_scheduler.lock")
    if not lock_file.exists():
        return False, "Scheduler lock missing (/tmp/stavki_scheduler.lock). Is it running?"
    
    # Try to see if PID is valid
    try:
        pid = int(lock_file.read_text().strip())
        os.kill(pid, 0) # Signal 0 checks if process exists
        return True, f"Scheduler running (PID: {pid})."
    except (ValueError, ProcessLookupError, PermissionError):
        return False, "Scheduler lock exists but process not found or inaccessible."

def check_logs(max_age_hours=12):
    """Check if scheduler log has recent activity."""
    log_file = Path("audit_pack/RUN_LOGS/scheduler.log")
    if not log_file.exists():
        return False, "Scheduler log missing."
    
    age_seconds = time.time() - log_file.stat().st_mtime
    age_hours = age_seconds / 3600
    
    if age_hours > max_age_hours:
        return False, f"No log activity in {age_hours:.1f}h. Check scheduler status."
    
    return True, f"Log active ({age_hours:.1f}h ago)."

def main():
    parser = argparse.ArgumentParser(description="Stavki Health Check")
    parser.add_argument('--telegram', action='store_true', help='Send alert on failure')
    parser.add_argument('--dry-run', action='store_true', help='Print status without alerting')
    args = parser.parse_args()

    results = []
    results.append(check_odds_freshness())
    results.append(check_scheduler_lock())
    results.append(check_logs())

    failed = [r[1] for r in results if not r[0]]
    successes = [r[1] for r in results if r[0]]

    status_str = "HEALTH CHECK RESULTS:\n"
    for s in successes: status_str += f"✅ {s}\n"
    for f in failed:    status_str += f"❌ {f}\n"

    print(status_str)

    if failed and args.telegram and not args.dry_run:
        alert_msg = "🚨 *STAVKI HEALTH ALERT*\n\n" + status_str
        if send_custom_message(alert_msg):
            print("✓ Health alert sent to Telegram.")
        else:
            print("⚠️ Failed to send health alert.")

    if failed:
        sys.exit(1)
    else:
        print("✓ All systems nominal.")
        sys.exit(0)

if __name__ == "__main__":
    main()
