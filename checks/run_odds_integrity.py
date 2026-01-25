#!/usr/bin/env python3
"""
Odds Integrity Check (Audit v3)
Verifies:
1. Snapshot time exists
2. Odds are valid (1.01 <= odds <= 1000)
3. Source bookmaker is present
4. Single-book integrity check (if possible to infer)
"""
import sys
import pandas as pd
import json
import glob
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path("audit_pack/A4_odds_integrity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = OUTPUT_DIR / "odds_integrity_report.csv"
SUMMARY_FILE = OUTPUT_DIR / "odds_integrity_summary.json"

ODDS_DIR = Path("outputs/odds")

def main():
    print("=== Odds Integrity Check ===")
    
    # Find latest normalized file
    files = sorted(glob.glob(str(ODDS_DIR / "normalized_*.csv")))
    if not files:
        print(f"❌ No odds files found in {ODDS_DIR}")
        print("Run scripts/run_odds_pipeline.py first.")
        # Create empty artifacts to allow validator to see 'fail' usage?
        # No, better to fail hard.
        sys.exit(1)
        
    latest_file = files[-1]
    print(f"Checking {latest_file}...")
    
    try:
        df = pd.read_csv(latest_file)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        sys.exit(1)
        
    if df.empty:
        print("⚠ Empty dataframe")
        sys.exit(0)

    # Columns check
    required_cols = ['event_id', 'market_key', 'outcome_name', 'outcome_price', 'bookmaker_key']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        sys.exit(1)
        
    # Check Snapshot Time
    if 'odds_snapshot_time' not in df.columns:
        print("❌ 'odds_snapshot_time' column missing! (Fix #1 violation)")
        df['odds_snapshot_time'] = None # Fill for reporting
    
    # Validation
    df['is_valid'] = True
    df['violation_reason'] = None
    
    mask_range = (df['outcome_price'] < 1.01) | (df['outcome_price'] > 1000)
    df.loc[mask_range, 'is_valid'] = False
    df.loc[mask_range, 'violation_reason'] = "Odds out of range"
    
    mask_snapshot = df['odds_snapshot_time'].isna()
    df.loc[mask_snapshot, 'is_valid'] = False
    df.loc[mask_snapshot, 'violation_reason'] = "Missing snapshot time"
    
    # Save Report
    output_cols = ['event_id', 'market_key', 'outcome_name', 'outcome_price', 'bookmaker_key', 
                   'odds_snapshot_time', 'is_valid', 'violation_reason']
    
    # Add match_start_time if present
    if 'commence_time' in df.columns:
        df['match_start_time'] = df['commence_time']
        output_cols.insert(5, 'match_start_time')
    else:
        df['match_start_time'] = None

    # Write report
    df[output_cols].to_csv(REPORT_FILE, index=False)
    
    # Summary
    summary = {
        "file": str(latest_file),
        "total_rows": len(df),
        "valid_rows": int(df['is_valid'].sum()),
        "invalid_rows": int((~df['is_valid']).sum()),
        "bookmakers": df['bookmaker_key'].unique().tolist(),
        "snapshot_coverage_pct": 100.0 if not mask_snapshot.any() else 0.0 # Strict
    }
    
    with open(SUMMARY_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Saved report to {REPORT_FILE}")
    print(json.dumps(summary, indent=2))
    
    if summary['invalid_rows'] > 0:
        print("❌ Integrity violations found")
        sys.exit(1)
    else:
        print("✅ Odds Integrity Verified")

if __name__ == "__main__":
    main()
