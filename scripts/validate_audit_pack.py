import os
import sys
import pandas as pd
import json

REQUIRED_FILES = [
    'audit_pack/A4_odds_integrity/odds_integrity_report.csv',
    'audit_pack/A6_metrics/metrics_summary.json',
    'audit_pack/A7_backtest/bets_backtest.csv',
    'audit_pack/A7_backtest/backtest_summary.json',
    'audit_pack/A8_staking/stake_cap_check.csv',
    'audit_pack/A9_live/predictions.csv',
    'audit_pack/A9_live/alerts_sent.csv',
    'audit_pack/A9_live/run_scheduler_log.txt',
    'audit_pack/RUN_LOGS/01_install.log'
]

REQUIRED_IMAGES = [
    'audit_pack/A6_metrics/calibration_plot.png',
    'audit_pack/A6_metrics/reliability_curve.png',
    'audit_pack/A6_metrics/probability_hist.png',
    'audit_pack/A7_backtest/equity_curve.png',
    'audit_pack/A7_backtest/drawdown_curve.png'
]

def validate():
    print("=== Starting Audit Pack Validation ===")
    failed = False
    
    # 1. Check Files Existence & Size
    for f in REQUIRED_FILES + REQUIRED_IMAGES:
        if not os.path.exists(f):
            print(f"❌ MISSING: {f}")
            failed = True
        elif os.path.getsize(f) == 0:
            print(f"❌ EMPTY: {f}")
            failed = True
        else:
            print(f"✅ FOUND: {f}")

    # 2. Check CSV Readability
    for f in REQUIRED_FILES:
        if f.endswith('.csv') and os.path.exists(f) and os.path.getsize(f) > 0:
            try:
                df = pd.read_csv(f)
                if df.empty:
                    print(f"⚠️  WARNING: CSV is valid but has 0 rows: {f}")
                else:
                    print(f"✅ CSV VALID ({len(df)} rows): {f}")
            except Exception as e:
                print(f"❌ CSV BROKEN: {f} -> {e}")
                failed = True

    # 3. Check JSON Validity
    for f in REQUIRED_FILES:
        if f.endswith('.json') and os.path.exists(f) and os.path.getsize(f) > 0:
            try:
                with open(f, 'r') as jf:
                    json.load(jf)
                print(f"✅ JSON VALID: {f}")
            except Exception as e:
                print(f"❌ JSON BROKEN: {f} -> {e}")
                failed = True

    if failed:
        print("\n⛔ VALIDATION FAILED")
        sys.exit(1)
    else:
        print("\n🎉 VALIDATION PASSED")
        sys.exit(0)

if __name__ == "__main__":
    validate()
