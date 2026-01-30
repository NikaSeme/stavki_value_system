#!/usr/bin/env python3
"""
STAVKI Backtest Simulator
-------------------------
Replays historical odds snapshots to test "Smart Blending" strategies.
Answers the question: "Who should we trust? Model or Market?"

Usage:
    python scripts/backtest_simulator.py --days 7
"""

import os
import sys
import glob
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy.value_live import (
    get_model_probabilities,
    compute_ev_candidates,
    initialize_ml_model
)
from src.strategy.blending import get_blending_alpha

def load_snapshots(data_dir="outputs/odds"):
    """Load all raw_master JSON snapshots."""
    pattern = f"{data_dir}/raw_master_*.json"
    files = sorted(glob.glob(pattern))
    print(f"📚 Found {len(files)} snapshots in {data_dir}")
    return files

def simulate_strategy(events, odds_df, profile_name, model_weight, ev_threshold=0.05):
    """
    Run EV finding on a set of events with a specific Model Weight (Alpha).
    """
    # 1. Get Model Probs (Pure)
    try:
        model_probs, s_data = get_model_probabilities(events, odds_df, model_type='ml')
        # DEBUG DIAGNOSTICS
        if len(model_probs) > 0:
            print(f"    - Model generated probs for {len(model_probs)} events")
            first_key = list(model_probs.keys())[0]
            print(f"    - Example Probs: {model_probs[first_key]}")
        else:
            print(f"    - Model generated 0 probabilities (Check Feature Extractor!)")
    except Exception as e:
        print(f"    - Model Error: {e}")
        return []

    # 2. Get Market Best Prices
    # (Simplified: assume odds_df has 'odds' column for simulation)
    # in real pipeline we use select_best_prices. Here we reuse odds_df which should be processed.
    # Actually, odds_df in raw_master might be complex. 
    # Let's assume we can map it.
    
    # We need 'best_prices' DF format for compute_ev_candidates
    # If the input odds_df is already the 'events_latest.csv' format, implies best prices.
    best_prices = odds_df.copy()
    
    # 3. Compute EV with specific Alpha
    candidates = compute_ev_candidates(
        model_probs,
        best_prices,
        sentiment_data=s_data,
        threshold=ev_threshold,
        alpha_shrink=model_weight, # THE KEY PARAMETER
        prob_sum_tol=0.05,
        drop_bad_prob_sum=False,
        renormalize=False,
        confirm_high_odds=False,
        high_odds_threshold=20.0,
        high_odds_p_threshold=0.0,
        cap_high_odds_prob=1.0, 
        bankroll=1000.0
    )
    
    return candidates

def main():
    print("🧪 STAVKI Backtest Simulator (V6 Alpha Engine)")
    print("="*60)
    
    # 1. Find Data (Use normalized CSVs which are easier to work with than raw JSON)
    # events_latest_*.csv contains the flattened structure we need.
    pattern = "outputs/odds/events_latest_*.csv"
    files = sorted(glob.glob(pattern))
    
    if not files:
        print("❌ No historical CSV data found in outputs/odds/")
        sys.exit(1)
        
    print(f"📅 Analyzing {len(files)} historical snapshots...")
    
    # Prepare Results Container
    # We will aggregate: Count of Bets, Avg EV, Avg Divergence
    results = {
        'Conservative (10% Model)': {'bets': 0, 'sum_ev': 0, 'sum_div': 0},
        'Balanced (50% Model)':     {'bets': 0, 'sum_ev': 0, 'sum_div': 0},
        'Aggressive (90% Model)':   {'bets': 0, 'sum_ev': 0, 'sum_div': 0}
    }
    
    # Initialize Model ONCE
    initialize_ml_model()
    
    # Iterate through snapshots (Limit to last 5 to save time in dev)
    for f_path in files[-5:]:
        print(f"  Processing {Path(f_path).name}...")
        try:
            df = pd.read_csv(f_path)
            if df.empty: continue
            
            # Filter for soccer
            df = df[df['sport_key'].str.contains('soccer')]
            
            # Normalize Columns if needed (ensure 'odds' exists)
            # events_latest usually has 'home_odds', 'draw_odds', 'away_odds' 
            # OR it might be in 'selection' format? 
            # compute_ev_candidates expects rows with 'odds', 'selection' (home/draw/away)
            # Actually compute_ev_candidates iterates through matches and checks 'home_team', 'away_team'.
            # It expects `best_prices` to have one row per outcome? No, usually one row per event?
            # Let's check `select_best_prices`. It melts the DF.
            # So we need to melt `df` if it's wide.
            
            # Quick Melt Simulation (ADAPTED FOR LONG FORMAT)
            # The CSV is already melted (Long Format): outcome_name, outcome_price
            
            # COPY columns instead of rename to preserve originals for get_model_probabilities
            check_df = df.copy()
            if 'outcome_name' in check_df.columns:
                 check_df['selection'] = check_df['outcome_name']
            if 'outcome_price' in check_df.columns:
                 check_df['odds'] = check_df['outcome_price']
            
            # Ensure required columns exist
            required_cols = ['event_id', 'sport_key', 'commence_time', 'home_team', 'away_team', 'selection', 'odds']
            for c in required_cols:
                if c not in check_df.columns:
                    # Attempt simple defaults if non-critical
                    if c == 'selection': check_df['selection'] = check_df.get('outcome_name')
                    if c == 'odds': check_df['odds'] = check_df.get('outcome_price')
            
            events_subset = df[['event_id', 'sport_key', 'home_team', 'away_team', 'commence_time']].drop_duplicates('event_id')

            # TIME TRAVEL HACK: Shift dates to future to bypass "past event" filters
            from datetime import timedelta
            now_sim = datetime.now()
            future_target = now_sim + timedelta(days=1)
            
            def shift_time(t_str):
                try:
                    # Parse original
                    # dt = datetime.fromisoformat(t_str.replace('Z', '+00:00'))
                    # Ignore original date, just force everything to "Tomorrow"
                    return future_target.isoformat()
                except:
                    return t_str

            check_df['commence_time'] = check_df['commence_time'].apply(shift_time)
            events_subset = events_subset.copy()
            events_subset['commence_time'] = events_subset['commence_time'].apply(shift_time)

            # --- RUN SIMULATIONS ---

            # --- RUN SIMULATIONS ---
            
            # Profile 1: Conservative (10%)
            c1 = simulate_strategy(events_subset, check_df, 'Conservative', 0.10)
            results['Conservative (10% Model)']['bets'] += len(c1)
            results['Conservative (10% Model)']['sum_ev'] += sum(x['ev_pct'] for x in c1)
            results['Conservative (10% Model)']['sum_div'] += sum(x['model_market_div'] for x in c1)

            # Profile 2: Balanced (50%)
            c2 = simulate_strategy(events_subset, check_df, 'Balanced', 0.50)
            results['Balanced (50% Model)']['bets'] += len(c2)
            results['Balanced (50% Model)']['sum_ev'] += sum(x['ev_pct'] for x in c2)
            results['Balanced (50% Model)']['sum_div'] += sum(x['model_market_div'] for x in c2)

            # Profile 3: Alpha Hunter (90%)
            c3 = simulate_strategy(events_subset, check_df, 'Aggressive', 0.90)
            results['Aggressive (90% Model)']['bets'] += len(c3)
            results['Aggressive (90% Model)']['sum_ev'] += sum(x['ev_pct'] for x in c3)
            results['Aggressive (90% Model)']['sum_div'] += sum(x['model_market_div'] for x in c3)

        except Exception as e:
            print(f"  ⚠️ Check failed: {e}")

    # Report
    print("\n📊 SIMULATION RESULTS (Last 5 Snapshots)")
    print(f"{'PROFILE':<25} | {'BETS':<6} | {'AVG EV':<10} | {'AVG DIV':<10} | {'NOTES'}")
    print("-" * 80)
    
    for name, data in results.items():
        count = data['bets']
        if count > 0:
            avg_ev = data['sum_ev'] / count
            avg_div = (data['sum_div'] / count) * 100
        else:
            avg_ev = 0; avg_div = 0
            
        note = "✅ Safe" if avg_ev < 10 else "⚠️ SUSPICIOUS" if avg_ev > 30 else "🤔 Aggressive"
        
        print(f"{name:<25} | {count:<6} | {avg_ev:.2f}%     | {avg_div:.1f}%      | {note}")
        
    print("\n💡 INTERPRETATION:")
    print("- 'Conservative': Mostly trusts Pinnacle. Low EV but high win rate.")
    print("- 'Aggressive': Trusts Model. If Avg EV > 30%, it is likely hallucinating.")
    print("- Your Goal: Find the spot where EV is 10-15% (Real Edge).")

if __name__ == "__main__":
    main()
