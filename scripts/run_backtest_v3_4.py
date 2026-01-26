#!/usr/bin/env python3
"""
Rigorous Value Betting Backtest (v3.4)
- Uses ONLY Out-of-Sample data (Val + Test)
- Reconstructs Bookmaker Odds with 5% Vig (Stress Test)
- Applies Strict Filters: High Odds (>10), Max Bets/Day, Kelly Capping
- Applies Betfair Commission (2%) to Net Winnings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from catboost import CatBoostClassifier
import json
from datetime import datetime

# --- CONFIG ---
INITIAL_BANKROLL = 1000.0
KELLY_FRACTION = 0.5
MAX_STAKE_PCT = 0.05
COMMISSION_RATE = 0.02 # Betfair
VIG_PCT = 0.05 # Assumed average margin 5%
HIGH_ODDS_THRESHOLD = 10.0
MAX_BETS_PER_LEAGUE_DAY = 2
MIN_EV_THRESHOLD = 0.02

def load_data():
    print("Loading OOS Data (Val + Test)...")
    val = pd.read_parquet("data/processed/splits_v3_3/val.parquet")
    test = pd.read_parquet("data/processed/splits_v3_3/test.parquet")
    
    # Combine OOS
    df = pd.concat([val, test]).sort_values("Date").reset_index(drop=True)
    print(f"Total OOS Rows: {len(df)}")
    return df

def reconstruct_odds(df):
    """
    Reconstruct Bookmaker Odds from No-Vig Market Probabilities.
    We assume the market prob is 'Fair', and Bookie adds VIG.
    Odds = 1 / (Prob_Fair * (1 + Vig))
    """
    # H
    df['OddsH'] = 1 / (df['MarketProbHomeNoVig'] * (1 + VIG_PCT))
    # D
    df['OddsD'] = 1 / (df['MarketProbDrawNoVig'] * (1 + VIG_PCT))
    # A
    df['OddsA'] = 1 / (df['MarketProbAwayNoVig'] * (1 + VIG_PCT))
    
    return df

def get_model_probs(model, df):
    model_features = model.feature_names_
    print(f"Model expects {len(model_features)} features: {model_features}")
    
    # Ensure all exist
    missing = [f for f in model_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in Data: {missing}")
        
    print(f"Predicting with features...")
    preds = model.predict_proba(df[model_features])
    
    # CatBoost MultiClass: [Pr(0), Pr(1), Pr(2)]?
    # Usually classes are inferred. If FTR is H, D, A.
    # We need to know class order.
    classes = model.classes_
    print(f"Model Classes: {classes}")
    
    # Map to H, D, A columns
    # Assume 0=A, 1=D, 2=H (Alphabetical?) OR H, D, A?
    # Usually matches targets.
    # Let's assume standard mapping: Home, Draw, Away if not specified? 
    # Actually, in soccer usually H=HomeWin, D=Draw, A=AwayWin.
    
    probs_df = pd.DataFrame(preds, columns=classes, index=df.index)
    return probs_df

def kelly_criterion(prob, odds, bankroll):
    if prob * odds <= 1:
        return 0.0
    
    f = (prob * odds - 1) / (odds - 1)
    stake = f * KELLY_FRACTION * bankroll
    cap = bankroll * MAX_STAKE_PCT
    return min(max(0, stake), cap)

def run_simulation(df, probs_df):
    bankroll = INITIAL_BANKROLL
    history = []
    bets_log = []
    
    # Group by Date to enforce daily limits
    df['DateStr'] = df['Date'].dt.date.astype(str)
    
    for date, group in df.groupby('DateStr'):
        dates_bets = []
        
        # 1. Find Candidates
        for idx, row in group.iterrows():
            # Check 3 outcomes: H, D, A
            # Outcome mapping: FTR 'H' means Home Win.
            
            # Home
            ph = probs_df.loc[idx, 'H'] if 'H' in probs_df.columns else 0
            od_h = row['OddsH']
            ev_h = ph * od_h - 1
            
            if ev_h > MIN_EV_THRESHOLD and od_h < HIGH_ODDS_THRESHOLD:
                dates_bets.append({
                    'idx': idx,
                    'selection': 'H',
                    'odds': od_h,
                    'prob': ph,
                    'ev': ev_h,
                    'result': 1 if row['FTR'] == 'H' else 0
                })
                
            # Away
            pa = probs_df.loc[idx, 'A'] if 'A' in probs_df.columns else 0
            od_a = row['OddsA']
            ev_a = pa * od_a - 1
            
            if ev_a > MIN_EV_THRESHOLD and od_a < HIGH_ODDS_THRESHOLD:
                dates_bets.append({
                    'idx': idx,
                    'selection': 'A',
                    'odds': od_a,
                    'prob': pa,
                    'ev': ev_a,
                    'result': 1 if row['FTR'] == 'A' else 0
                })
                
            # Draw
            pd_raw = probs_df.loc[idx, 'D'] if 'D' in probs_df.columns else 0
            od_d = row['OddsD']
            ev_d = pd_raw * od_d - 1
            
            if ev_d > MIN_EV_THRESHOLD and od_d < HIGH_ODDS_THRESHOLD:
                dates_bets.append({
                    'idx': idx,
                    'selection': 'D',
                    'odds': od_d,
                    'prob': pd_raw,
                    'ev': ev_d,
                    'result': 1 if row['FTR'] == 'D' else 0
                })
        
        # 2. Sort & Limit (Diversification)
        # Max 2 bets per "League" (Here assuming whole dataset is 1 league/day mix)
        # We enforce "Max 2 bets per day" as a proxy for league limit in this simplified backtest
        dates_bets.sort(key=lambda x: x['ev'], reverse=True)
        selected_bets = dates_bets[:MAX_BETS_PER_LEAGUE_DAY]
        
        # 3. Execute Bets
        daily_pnl = 0
        for bet in selected_bets:
            stake = kelly_criterion(bet['prob'], bet['odds'], bankroll)
            
            if stake < 1.0: continue # Min stake
            
            # Result
            gross_pnl = 0
            if bet['result'] == 1:
                profit = (bet['odds'] - 1) * stake
                # Commission
                commission = profit * COMMISSION_RATE
                gross_pnl = profit - commission
            else:
                gross_pnl = -stake
            
            daily_pnl += gross_pnl
            bankroll += gross_pnl
            
            bets_log.append({
                'date': date,
                'selection': bet['selection'],
                'odds': round(bet['odds'], 2),
                'prob': round(bet['prob'], 3),
                'ev': round(bet['ev'], 3),
                'stake': round(stake, 2),
                'result': 'WIN' if bet['result'] == 1 else 'LOSS',
                'pnl': round(gross_pnl, 2),
                'bankroll': round(bankroll, 2)
            })
            
        history.append({'date': date, 'bankroll': bankroll, 'bets_count': len(selected_bets)})
        
    return pd.DataFrame(history), pd.DataFrame(bets_log)

def main():
    # 1. Load Data
    df = load_data()
    df = reconstruct_odds(df)
    
    # 2. Load Model
    model = CatBoostClassifier()
    model.load_model("models/catboost_soccer_v3_3.cbm")
    
    # 3. Predict
    probs_df = get_model_probs(model, df)
    
    # 4. Simulate
    history_df, bets_df = run_simulation(df, probs_df)
    
    # 5. Report
    final_bankroll = history_df['bankroll'].iloc[-1]
    roi = (final_bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL
    
    print("="*60)
    print(f"BACKTEST RESULTS (v3.4)")
    print("="*60)
    print(f"Start Bankroll: {INITIAL_BANKROLL}")
    print(f"Final Bankroll: {final_bankroll:.2f}")
    print(f"Net Profit:     {final_bankroll - INITIAL_BANKROLL:.2f}")
    print(f"ROI:            {roi:.2%}")
    print(f"Total Bets:     {len(bets_df)}")
    
    # Save Artifacts
    out_dir = Path("audit_pack/A7_backtest")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    bets_df.to_csv(out_dir / "bets_backtest.csv", index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(history_df['date']), history_df['bankroll'])
    plt.title(f"Equity Curve v3.4 (Strict OOS + Fees)\nFinal: ${final_bankroll:.0f} (ROI {roi:.1%})")
    plt.xlabel("Date")
    plt.ylabel("Bankroll")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "equity_curve.png")
    print(f"Saved plot to {out_dir}/equity_curve.png")

if __name__ == "__main__":
    main()
