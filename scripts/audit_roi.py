#!/usr/bin/env python3
"""
ROI Audit Script
Goal: Compare ROI across different odds providers (Avg vs B365 vs Pinnacle vs Max)
to understand if -11% is due to "Soft" odds or model performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import logging
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.poisson_model import PoissonMatchPredictor
from src.strategy.league_router import LeagueRouter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_models(models_dir):
    # CatBoost Pipeline
    pipe_path = models_dir / 'catboost_pipeline_v1_latest.pkl'
    logger.info(f"Loading CatBoost Pipeline: {pipe_path}")
    cb_pipeline = joblib.load(pipe_path)
    
    # Poisson (Class)
    ps_path = models_dir / 'poisson_v1_latest.pkl'
    ps_model = PoissonMatchPredictor.load(ps_path)
        
    return cb_pipeline, ps_model

def get_poisson_probs(model, df):
    preds = model.predict(df)
    return preds[['prob_home', 'prob_draw', 'prob_away']].values

def calculate_roi(probs, odds_col, df, threshold=0.05):
    # Get odds array
    odds_map = {
        'Home': odds_col + 'H',
        'Draw': odds_col + 'D',
        'Away': odds_col + 'A'
    }
    
    # Check if cols exist (some might be B365H, others OddsH)
    # HACK: Handle common prefixes
    if odds_col == 'Odds':
        odds_map = {'Home': 'OddsHome', 'Draw': 'OddsDraw', 'Away': 'OddsAway'}
    elif odds_col == 'B365':
        odds_map = {'Home': 'B365H', 'Draw': 'B365D', 'Away': 'B365A'}
    elif odds_col == 'Pin':
        odds_map = {'Home': 'PSH', 'Draw': 'PSD', 'Away': 'PSA'} # Pinnacle often PS
    
    # Check existence
    if not all(c in df.columns for c in odds_map.values()):
        return None, 0
        
    odds = df[[odds_map['Home'], odds_map['Draw'], odds_map['Away']]].values
    
    # Expectation
    ev = (probs * odds) - 1
    
    # Calculate
    profit = 0.0
    staked = 0.0
    bets = 0
    
    y_true = df['FTR'].map({'H':0, 'D':1, 'A':2}).values
    
    for i in range(3):
        mask = ev[:, i] > threshold
        bets += mask.sum()
        staked += mask.sum()
        
        # Outcome
        won = (y_true == i)
        pnl = np.where(won[mask], odds[mask, i] - 1, -1.0)
        profit += pnl.sum()
        
    roi = (profit / staked * 100) if staked > 0 else 0.0
    return roi, bets

def main():
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / 'data' / 'processed' / 'multi_league_features_2021_2024.csv'
    models_dir = base_dir / 'models'
    
    logger.info("loading data...")
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Create FTR
    if 'FTR' not in df.columns:
        df['FTR'] = np.where(df['home_goals'] > df['away_goals'], 'H', 
                             np.where(df['home_goals'] < df['away_goals'], 'A', 'D'))
                             
    # Split Test Set (Last 15%)
    n = len(df)
    test_start = int(n * 0.85)
    test_df = df.iloc[test_start:].copy()
    logger.info(f"Test Set: {len(test_df)} matches")
    
    # Load Models & Predict
    cb_pipeline, ps_model = load_models(models_dir)
    
    # CatBoost Features
    with open(models_dir / 'metadata_v1_latest.json') as f:
        cb_cols = json.load(f)['features']
        
    # CatBoost Probs (Via Pipeline)
    # Ensure columns exist
    for c in cb_cols:
        if c not in test_df.columns: test_df[c] = 0
            
    p_cb = cb_pipeline.predict_proba(test_df[cb_cols])
    
    # Poisson Probs
    p_ps = get_poisson_probs(ps_model, test_df)
    
    # --- ROUTER APPLICATION ---
    router = LeagueRouter()
    
    # Initialize p_router with zeros
    p_router = np.zeros_like(p_cb)
    
    leagues = test_df['League'].unique()
    
    for league in leagues:
        mask = (test_df['League'] == league).values
        # Get weights from config
        w_c, w_n, w_p = router.get_weights(league)
        
        # Apply weights: w_c*CB + w_p*PS (Neural assumed 0 for now)
        p_router[mask] = w_c * p_cb[mask] + w_p * p_ps[mask]
        
    # Normalize rows
    sum_rows = p_router.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        p_router = p_router / sum_rows
        p_router = np.nan_to_num(p_router) # 0/0 -> 0 (No prediction)

    # Standard Global Ensemble
    # Just use equal weights for ref? Or defaults?
    # Let's use 0.5/0.5
    p_ens = 0.5 * p_cb + 0.5 * p_ps 
    sum_rows_ens = p_ens.sum(axis=1, keepdims=True)
    p_ens = p_ens / sum_rows_ens
    
    # --- COMPONENT DIAGNOSIS ---
    logger.info("\n--- COMPONENT DIAGNOSIS (Max Odds) ---")
    
    scenarios = [
        ("CatBoost Only", p_cb),
        ("Poisson Only", p_ps),
        ("Global 50/50", p_ens),
        ("SMART ROUTER", p_router)
    ]
    
    col_name = 'Max'
    
    for label, probs in scenarios:
        roi, bets = calculate_roi(probs, col_name, test_df)
        if roi is None:
             logger.info(f"{label}: Odds Column Missing")
        else:
             logger.info(f"{label}: ROI = {roi:.2f}% ({bets} bets)")

    # --- LEAGUE BREAKDOWN ---
    logger.info("\n--- LEAGUE BREAKDOWN (Model: SMART ROUTER) ---")
    
    for league in leagues:
        league_df = test_df[test_df['League'] == league]
        n_league = len(league_df)
        mask = (test_df['League'] == league).values
        
        logger.info(f"\n{league.upper()} ({n_league} matches):")
        
        # Show Router Performance vs Solo Components
        league_probs = p_router[mask]
        roi, bets = calculate_roi(league_probs, 'Max', league_df)
        
        # Compare to Solo CatBoost
        roi_cb, bets_cb = calculate_roi(p_cb[mask], 'Max', league_df)
        
        # Router Weights
        w_c, _, w_p = router.get_weights(league)
        
        logger.info(f"  Router (W_CB={w_c:.3f}): ROI {roi:>6.2f}% ({bets} bets)")
        logger.info(f"  Solo CatBoost      : ROI {roi_cb:>6.2f}% ({bets_cb} bets)")
        
    logger.info("\n--- AUDIT COMPLETE ---")
    
if __name__ == "__main__":
    main()
