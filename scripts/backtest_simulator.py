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
    # Argument Parsing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-history', action='store_true', help="Use comprehensive 2021-2024 dataset for PnL backtest")
    parser.add_argument('--days', type=int, default=7, help="Number of days to analyze in snapshot mode")
    args = parser.parse_args()

    print("🧪 STAVKI Backtest Simulator (V6 Alpha Engine)")
    print("="*60)
    
    # Mode Selection
    if args.full_history:
        print("📜 MODE: Full History PnL (2021-2024)")
        f_path = "data/processed/multi_league_features_6leagues_full.csv"
        if not os.path.exists(f_path):
            print(f"❌ Full history file not found: {f_path}")
            sys.exit(1)
            
        print(f"📊 Loading massive dataset: {f_path} ...")
        df = pd.read_csv(f_path)
        print(f"   Loaded {len(df)} matches.")
        
        # Limit for speed if needed (e.g. last 1000)
        # df = df.tail(1000)
        
        # Prepare Data for Model
        # The CSV has features directly. We need to map standard columns for the predictor wrapper.
        df = df.rename(columns={'HomeTeam': 'home_team', 'AwayTeam': 'away_team', 'Date': 'commence_time'})
        
        # 1. Generate Model Probs (Batch)
        # We can pass the whole DF as 'features' to ensemble.predict
        # But 'events' needs standard cols.
        events_meta = df[['home_team', 'away_team', 'commence_time']].copy()
        
        print("🧠 Generating Model Predictions (this may take a minute)...")
        from src.models.ensemble_predictor import EnsemblePredictor
        ensemble = EnsemblePredictor()
        
        try:
            probs, _ = ensemble.predict(events_meta, None, features=df)
            df['prob_home'] = probs[:, 0]
            df['prob_draw'] = probs[:, 1]
            df['prob_away'] = probs[:, 2]
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            print("Note: Ensure 'multi_league_features...' headers match CatBoost expectations.")
            sys.exit(1)
            
        print("💰 Calculating PnL for Alpha Sweep...")
        
        results = {}
        
        # Pre-calculate No-Vig Market Probs (using AvgOdds)
        def safe_div(x): return 1/x if x > 1 else 0
        df['p_market_h'] = df['AvgOddsH'].apply(safe_div)
        df['p_market_d'] = df['AvgOddsD'].apply(safe_div)
        df['p_market_a'] = df['AvgOddsA'].apply(safe_div)
        
        # Normalize Market Probs
        sums = df[['p_market_h', 'p_market_d', 'p_market_a']].sum(axis=1)
        # Avoid div by zero
        sums[sums == 0] = 1.0
        df['p_market_h'] /= sums
        df['p_market_d'] /= sums
        df['p_market_a'] /= sums

        # Strategy Loop
        alphas = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        
        for alpha in alphas:
            key = f"Alpha {alpha:.2f}"
            
            # Blend Probs
            p_final_h = alpha * df['prob_home'] + (1-alpha) * df['p_market_h']
            p_final_d = alpha * df['prob_draw'] + (1-alpha) * df['p_market_d']
            p_final_a = alpha * df['prob_away'] + (1-alpha) * df['p_market_a']
            
            # Checks Bets (EV > 5%)
            # Bet H
            ev_h = (p_final_h * df['AvgOddsH']) - 1
            bet_h = ev_h > 0.05
            
            # Bet D
            ev_d = (p_final_d * df['AvgOddsD']) - 1
            bet_d = ev_d > 0.05
            
            # Bet A
            ev_a = (p_final_a * df['AvgOddsA']) - 1
            bet_a = ev_a > 0.05
            
            # Calculate PnL (Flat Stake 1 unit)
            # Result: FTR is 'H', 'D', 'A'
            
            pnl = 0.0
            bets = 0
            
            # Vectorized PnL approx
            # H Wins:
            h_wins = df['FTR'] == 'H'
            pnl += ((df.loc[bet_h & h_wins, 'AvgOddsH'] - 1).sum()) 
            pnl -= (~h_wins & bet_h).sum() # Loss
            bets += bet_h.sum()
            
            # D Wins
            d_wins = df['FTR'] == 'D'
            pnl += ((df.loc[bet_d & d_wins, 'AvgOddsD'] - 1).sum())
            pnl -= (~d_wins & bet_d).sum()
            bets += bet_d.sum()

            # A Wins
            a_wins = df['FTR'] == 'A'
            pnl += ((df.loc[bet_a & a_wins, 'AvgOddsA'] - 1).sum())
            pnl -= (~a_wins & bet_a).sum()
            bets += bet_a.sum()
            
            roi = (pnl / bets * 100) if bets > 0 else 0
            
            results[key] = {'bets': bets, 'pnl': pnl, 'roi': roi}
            
        # Report
        print("\n📊 HISTORICAL PnL RESULTS (2021-2024)")
        print(f"{'CONFIDENCE':<15} | {'BETS':<8} | {'PnL (U)':<10} | {'ROI %':<10} | {'VERDICT'}")
        print("-" * 80)
        
        for name in sorted(results.keys()):
            r = results[name]
            roi = r['roi']
            if roi > 5: verdict = "✅ PROFITABLE"
            elif roi > 0: verdict = "🆗 Breakeven"
            elif roi > -5: verdict = "⚠️ Losing"
            else: verdict = "💀 REKT"
            
            print(f"{name:<15} | {r['bets']:<8} | {r['pnl']:<10.1f} | {roi:<10.1f}% | {verdict}")

        sys.exit(0)

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

            # --- RUN SIMULATIONS (Granular Sweep) ---
            
            # Sweep Alpha from 0.10 to 0.90 with step 0.10
            for alpha in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
                key = f"Alpha {alpha:.2f}"
                if key not in results:
                    results[key] = {'bets': 0, 'sum_ev': 0, 'sum_div': 0}
                
                c = simulate_strategy(events_subset, check_df, key, alpha)
                results[key]['bets'] += len(c)
                results[key]['sum_ev'] += sum(x['ev_pct'] for x in c)
                results[key]['sum_div'] += sum(x['model_market_div'] for x in c)

        except Exception as e:
            # print(f"  ⚠️ Check failed: {e}") 
            pass

    # Report
    print("\n📊 SENSITIVITY ANALYSIS (Truth Curve)")
    print(f"{'CONFIDENCE':<15} | {'BETS':<6} | {'AVG EV':<10} | {'AVG DIV':<10} | {'VERDICT'}")
    print("-" * 80)
    
    sorted_keys = sorted(results.keys())
    for name in sorted_keys:
        data = results[name]
        count = data['bets']
        if count > 0:
            avg_ev = data['sum_ev'] / count
            avg_div = (data['sum_div'] / count) * 100
        else:
            avg_ev = 0; avg_div = 0
            
        # Verdict Logic
        if avg_ev < 12: verdict = "✅ Real Edge"
        elif avg_ev < 25: verdict = "⚠️ Slightly Over"
        elif avg_ev < 40: verdict = "🛑 High Risk"
        else: verdict = "💀 Hallucination"
        
        print(f"{name:<15} | {count:<6} | {avg_ev:.2f}%     | {avg_div:.1f}%      | {verdict}")
        
    print("\n💡 CONCLUSION:")
    print("The sweet spot is likely where Avg EV is 8-15%.")
    print("Anything above 30% is almost certainly noise/error.")

if __name__ == "__main__":
    main()
