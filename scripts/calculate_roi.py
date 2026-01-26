#!/usr/bin/env python3
"""
Stavki ROI Calculator (V5)

Parses alerts_sent.csv and calculates potential and realized ROI.
Currently focuses on exposure and expected value since automated result fetching 
is external.

Usage:
    python scripts/calculate_roi.py
"""

import pandas as pd
from pathlib import Path

def main():
    print("=== ROI CALCULATOR (V5) ===")
    
    # Load Alerts Archive
    alerts_file = Path("audit_pack/A9_live/alerts_sent.csv")
    if not alerts_file.exists():
        print("❌ No alerts found (alerts_sent.csv missing).")
        return
        
    try:
        df = pd.read_csv(alerts_file)
        print(f"Loaded {len(df)} alerts.")
    except Exception as e:
        print(f"❌ Error reading csv: {e}")
        return
        
    if df.empty:
        print("No bets to analyze.")
        return

    # Basic Stats
    total_bets = len(df)
    
    # Ensure columns exist
    if 'ev_pct' not in df.columns:
        print("❌ Missing 'ev_pct' column.")
        return
        
    avg_ev = df['ev_pct'].mean()
    avg_odds = df['odds'].mean()
    
    # Stake Analysis
    # Should use 'stake' (which is cash) or 'stake_pct'
    # Assuming standard bankroll 1000 if 'stake' is missing? 
    # V5 logs 'stake' (cash) and 'stake_pct'.
    
    total_exposure = 0.0
    if 'stake' in df.columns:
        total_exposure = df['stake'].sum()
    
    print("-" * 30)
    print(f"Total Bets:       {total_bets}")
    print(f"Average Odds:     {avg_odds:.2f}")
    print(f"Average EV:       +{avg_ev:.2f}%")
    print(f"Total Exposure:   £{total_exposure:.2f}")
    print("-" * 30)
    
    # ROI simulation (Expected)
    # Expected Profit = Sum(Stake * EV_decimal)
    # EV_decimal is roughly ev_pct/100 * Stake? 
    # Actually EV in % is (Prob*Odds - 1). 
    # So Expected Profit = Stake * (Prob*Odds - 1)
    
    if 'stake' in df.columns:
        # Reconstruct prob? 'p_final' or 'p_model'
        # Or just use ev_pct directly if accurate
        # ev_pct is usually ((Prob*Odds)-1)*100
        # So EV_decimal = ev_pct / 100
        
        expected_profit = (df['stake'] * (df['ev_pct'] / 100.0)).sum()
        expected_roi = (expected_profit / total_exposure * 100) if total_exposure > 0 else 0
        
        print(f"Expected Profit:  £{expected_profit:.2f}")
        print(f"Expected ROI:     {expected_roi:.2f}%")
        
    print("-" * 30)
    print("Note: Realized ROI requires 'result' column (Win/Loss).")
    print("      Merge with external results to compute.")

if __name__ == "__main__":
    main()
