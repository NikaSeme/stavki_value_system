#!/usr/bin/env python3
"""
Diagnostic: Generate Manual EV Check (A6)
Calculates Manual EV for top 3 live alerts to verify math.
"""
import pandas as pd
import json
import glob
import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

OUTPUT_DIR = Path('audit_pack/A6_metrics')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANUAL_EV_FILE = OUTPUT_DIR / 'manual_ev_check.csv'

def main():
    print("=== Generating Manual EV Check Table ===")
    
    # Needs predictions.csv (populated by run_value_finder.py)
    preds_file = Path('audit_pack/A9_live/predictions.csv')
    
    if not preds_file.exists():
        print(f"❌ {preds_file} not found. Cannot generate manual check.")
        # Create empty
        pd.DataFrame(columns=['selection', 'odds', 'p_model', 'p_implied', 'ev_calc', 'ev_reported', 'diff']).to_csv(MANUAL_EV_FILE, index=False)
        sys.exit(0)
        
    df = pd.read_csv(preds_file)
    
    if df.empty:
         print("⚠ Predictions empty.")
         sys.exit(0)

    # Take top 3 by EV
    if 'ev' in df.columns:
        top_3 = df.nlargest(3, 'ev')
    else:
        top_3 = df.head(3)
        
    check_rows = []
    
    for _, row in top_3.iterrows():
        sel = row['selection']
        odds = float(row['odds'])
        p_model = float(row['p_model'])
        ev_reported = float(row['ev'])
        
        # Manual Calc
        # EV = (p_model * odds) - 1
        ev_calc = (p_model * odds) - 1
        
        # Implied
        p_implied = 1 / odds
        
        diff = abs(ev_calc - ev_reported)
        
        check_rows.append({
            'selection': sel,
            'odds': odds,
            'p_model': p_model,
            'p_implied': round(p_implied, 4),
            'ev_calc': round(ev_calc, 4),
            'ev_reported': round(ev_reported, 4),
            'diff': round(diff, 6),
            'status': 'PASS' if diff < 0.0001 else 'FAIL'
        })
        
    res_df = pd.DataFrame(check_rows)
    res_df.to_csv(MANUAL_EV_FILE, index=False)
    print(f"✅ Generated {MANUAL_EV_FILE}")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()
