#!/usr/bin/env python3
"""
STAVKI Cleanup & Maintenance (M32)

Retention Policy:
1. outputs/odds/*.csv -> Delete files older than N days.
2. audit_pack/A9_live/*.csv -> Delete files older than N days (except critical ones).
3. logs/*.log -> Basic rotation if needed.

Usage:
    python scripts/cleanup_maintenance.py --days 30
    python scripts/cleanup_maintenance.py --dry-run
"""

import os
import time
import argparse
from pathlib import Path

def cleanup_directory(directory_path, days_to_keep, pattern="*", excludes=None):
    """Delete files older than N days in a directory."""
    path = Path(directory_path)
    if not path.exists():
        print(f"Directory missing: {path}")
        return 0
    
    cutoff = time.time() - (days_to_keep * 86400)
    deleted_count = 0
    
    if excludes is None: excludes = []
    
    for f in path.glob(pattern):
        if f.is_file() and f.stat().st_mtime < cutoff:
            if f.name in excludes:
                continue
            
            try:
                # print(f"Deleting stale file: {f.name}")
                f.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {f}: {e}")
                
    return deleted_count

def main():
    parser = argparse.ArgumentParser(description="Stavki Maintenance & Cleanup")
    parser.add_argument('--days', type=int, default=30, help='Retention days (default: 30)')
    parser.add_argument('--dry-run', action='store_true', help='Scan without deleting')
    args = parser.parse_args()

    print(f"Running maintenance (Retention: {args.days} days)...")
    
    # Target 1: Odds Data
    # Path outputs/odds/
    odds_path = "outputs/odds"
    
    # Target 2: Audit live artifacts
    # Path audit_pack/A9_live/
    audit_path = "audit_pack/A9_live"

    total_deleted = 0
    
    targets = [
        (odds_path, "events_latest_*.csv"),
        (audit_path, "predictions_*.csv"),
        (audit_path, "meta_selections_*.csv"),
        (audit_path, "alerts_sent.csv"), # Maybe keep this longer or rotate? Actually alerts_sent is a log.
    ]
    
    for dir_path, pat in targets:
        if args.dry_run:
            # Simple scan
            p = Path(dir_path)
            if p.exists():
                old_files = [f for f in p.glob(pat) if f.is_file() and f.stat().st_mtime < (time.time() - args.days * 86400)]
                print(f"  [Dry Run] {dir_path}/{pat}: found {len(old_files)} stale files.")
        else:
            deleted = cleanup_directory(dir_path, args.days, pattern=pat)
            print(f"  ✓ {dir_path}/{pat}: deleted {deleted} stale files.")
            total_deleted += deleted

    print(f"Maintenance complete. Total files removed: {total_deleted}")

if __name__ == "__main__":
    main()
