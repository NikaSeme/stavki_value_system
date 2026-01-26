import os
import sys
import pandas as pd
import json

REQUIRED_FILES = [
    'audit_pack/A9_live/model_load_report.json',
    'audit_pack/A9_live/predictions.csv',
    'audit_pack/A9_live/alerts_sent.csv',
    'audit_pack/A9_live/top_ev_bets.csv',
    'audit_pack/A9_live/selection_report.json',
    'audit_pack/A9_live/telegram_sender_manifest.json',
    'audit_pack/A8_staking/stake_cap_check.csv',
    'audit_pack/A7_backtest/bets_backtest.csv',
    'audit_pack/A6_metrics/catboost_metrics.json',
    'audit_pack/A6_metrics/hparam_search_report.json',
    'audit_pack/A5_models/models_manifest.json',
    'audit_pack/A3_time_integrity/time_split_report_v3_3.json',
    'audit_pack/A2_data/events_schema.md'
]

REQUIRED_IMAGES = [
    'audit_pack/A6_metrics/calibration_plot_catboost.png',
    'audit_pack/A7_backtest/equity_curve.png'
]

def validate():
    print("=== Starting Audit Pack Validation (v3.4) ===")
    failed = False
    
    # 1. Check Files Existence & Size
    for f in REQUIRED_FILES + REQUIRED_IMAGES:
        if not os.path.exists(f):
            # A2 schemas might be in generic md, assuming path correct
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
                    
                # v3.4 Logic: Check HIGH ODDS Filter in top_ev_bets
                if 'top_ev_bets.csv' in f:
                    if 'odds' in df.columns:
                        max_odds = df['odds'].max()
                        if max_odds > 10.0:
                            print(f"❌ LOGIC FAIL: Found odds > 10.0 in {f} (Max: {max_odds})")
                            failed = True
                        else:
                            print(f"✅ LOGIC PASS: All odds <= 10.0 in Top Bets")
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
