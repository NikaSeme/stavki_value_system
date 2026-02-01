
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.poisson_model import PoissonMatchPredictor
from src.logging_setup import get_logger

logger = get_logger(__name__)

def calculate_roi(probs, odds_col, df):
    # Map odds
    if odds_col == 'Max':
        odds = df[['MaxH', 'MaxD', 'MaxA']].values
    else:
        odds = df[['OddsHome', 'OddsDraw', 'OddsAway']].values # Avg
        
    ev = (probs * odds) - 1
    
    staked = 0.0
    profit = 0.0
    bets = 0
    
    y_true = df['FTR'].map({'H':0, 'D':1, 'A':2}).values
    
    for i in range(3):
        mask = ev[:, i] > 0.05
        bets += mask.sum()
        staked += mask.sum()
        
        won = (y_true == i)
        # Handle nan odds if any
        # Assuming odds clean
        pnl = np.where(won[mask], odds[mask, i] - 1, -1.0)
        profit += pnl.sum()
        
    roi = (profit / staked * 100) if staked > 0 else 0.0
    return roi, bets, profit

def main():
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / 'data' / 'processed' / 'multi_league_features_2021_2024.csv'
    model_path = base_dir / 'models' / 'poisson_v1_latest.pkl'
    
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    if 'FTR' not in df.columns:
         df['FTR'] = np.where(df['home_goals'] > df['away_goals'], 'H', 
                              np.where(df['home_goals'] < df['away_goals'], 'A', 'D'))
    
    # Split Test Set (Same as Audit)
    n = len(df)
    test_start = int(n * 0.85)
    test_df = df.iloc[test_start:].copy()
    
    logger.info(f"Test Set: {len(test_df)} matches ({test_df['Date'].min()} - {test_df['Date'].max()})")
    
    # 1. Static Baseline
    logger.info("--- STATIC POISSON ---")
    with open(model_path, 'rb') as f:
        # Load params but we need the class to reconstruct? 
        # Actually our class has a .load() method now
        pass
    
    static_model = PoissonMatchPredictor.load(model_path)
    static_preds = static_model.predict(test_df)
    
    p_static = static_preds[['prob_home', 'prob_draw', 'prob_away']].values
    roi_static, bets_static, _ = calculate_roi(p_static, 'Max', test_df)
    logger.info(f"Static ROI: {roi_static:.2f}% ({bets_static} bets)")
    
    # 2. Dynamic Rolling
    logger.info("\n--- DYNAMIC ROLLING POISSON ---")
    dynamic_model = PoissonMatchPredictor.load(model_path) # Start with same base state
    
    predictions = []
    
    # Sort chronologically essential
    test_df = test_df.sort_values('Date')
    
    total_matches = len(test_df)
    
    for i, (idx, row) in enumerate(test_df.iterrows()):
        if i % 100 == 0: sum_w = 0 # simple prog bar
        
        # 1. Predict (Before seeing result)
        probs = dynamic_model.predict_match(row['HomeTeam'], row['AwayTeam'], league=row['League'])
        predictions.append([probs['prob_home'], probs['prob_draw'], probs['prob_away']])
        
        # 2. Update (After seeing result)
        dynamic_model.update_match(
            home=row['HomeTeam'],
            away=row['AwayTeam'],
            hg=row['FTHG'],
            ag=row['FTAG'],
            league=row['League']
        )
        
    p_dynamic = np.array(predictions)
    roi_dyn, bets_dyn, profit_dyn = calculate_roi(p_dynamic, 'Max', test_df)
    
    logger.info(f"Dynamic ROI: {roi_dyn:.2f}% ({bets_dyn} bets)")
    logger.info(f"Improvement: {roi_dyn - roi_static:+.2f}%")
    
    # League Breakdown for Dynamic
    logger.info("\n--- DYNAMIC LEAGUE BREAKDOWN ---")
    leagues = test_df['League'].unique()
    
    for league in leagues:
        mask = (test_df['League'] == league).values
        roi, bets, _ = calculate_roi(p_dynamic[mask], 'Max', test_df[mask])
        st_roi, _, _ = calculate_roi(p_static[mask], 'Max', test_df[mask])
        logger.info(f"{league.upper():<12}: Static={st_roi:>6.2f}% -> Dynamic={roi:>6.2f}% (Diff: {roi-st_roi:+.2f}%)")
        
if __name__ == "__main__":
    main()
