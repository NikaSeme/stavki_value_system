#!/usr/bin/env python3
"""
Precision Per-League Optimizer
Finds the exact optimal weight mix (CatBoost + Poisson) for each league.
Granularity: 0.1% (step=0.001).
Output: updates models/league_config.json
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import logging
from datetime import datetime
from sklearn.metrics import log_loss
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy.league_router import LeagueRouter
from src.models.poisson_model import PoissonMatchPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Loaders ---
def load_catboost_pipeline(models_dir):
    pipe_path = models_dir / 'catboost_pipeline_v1_latest.pkl'
    logger.info(f"Loading CatBoost Pipeline: {pipe_path}")
    return joblib.load(pipe_path)

def load_poisson(models_dir):
    path = models_dir / 'poisson_v1_latest.pkl'
    logger.info(f"Loading Poisson: {path}")
    return PoissonMatchPredictor.load(path)

# --- Predictors ---
def get_pipeline_probs(pipeline, df, feature_names):
    X = df[feature_names].copy()
    for c in feature_names:
        if c not in X.columns: X[c] = 0
    return pipeline.predict_proba(X)

def get_poisson_probs(model, df):
    preds = model.predict(df)
    return preds[['prob_home', 'prob_draw', 'prob_away']].values

def calculate_roi(probs, odds, y_true):
    ev = (probs * odds) - 1
    staked = 0.0
    profit = 0.0
    bets = 0
    
    # Vectorized loop
    for i in range(3):
        mask = ev[:, i] > 0.05
        bets += mask.sum()
        staked += mask.sum()
        won = (y_true == i)
        pnl = np.where(won[mask], odds[mask, i] - 1, -1.0)
        profit += pnl.sum()
        
    return (profit / staked * 100) if staked > 0 else 0.0, bets

# --- Optimizer ---
def optimize_weights_1d(y_true, p_cb, p_ps, odds, step=0.001):
    """
    Find best w_c to maximize ROI.
    """
    best_roi = -float('inf')
    best_w = 1.0 # Default to CatBoost if fail
    
    # 0.000 to 1.000
    steps = np.arange(0, 1.0 + step, step)
    
    for w_c in steps:
        w_p = 1.0 - w_c
        
        # Mix
        p_mix = w_c * p_cb + w_p * p_ps
        
        # Norm
        s = p_mix.sum(axis=1, keepdims=True)
        p_mix = np.divide(p_mix, s, out=np.zeros_like(p_mix), where=s!=0)
        
        # ROI
        roi, bets = calculate_roi(p_mix, odds, y_true)
        
        if bets >= 10: # Min sample size
            if roi > best_roi:
                best_roi = roi
                best_w = w_c
    
    if best_roi == -float('inf'):
         logger.warning("No configuration found with >10 bets. Defaulting to CB=1.0")
         return 1.0, 0.0, 0.0
         
    return best_w, (1.0 - best_w), best_roi

def main():
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / 'data' / 'processed' / 'multi_league_features_2021_2024.csv'
    models_dir = base_dir / 'models'
    config_path = models_dir / 'league_config.json'
    
    # 1. Load Data
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    if 'FTR' not in df.columns:
        df['FTR'] = np.where(df['home_goals'] > df['away_goals'], 'H', 
                             np.where(df['home_goals'] < df['away_goals'], 'A', 'D'))
    y = df['FTR'].map({'H':0, 'D':1, 'A':2}).values
    
    # 2. Split
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85) # Validation Set is optimization target
    
    val_df = df.iloc[train_end:val_end].copy()
    y_val = y[train_end:val_end]
    
    logger.info(f"Optimization Set (Validation): {len(val_df)} matches")
    
    # Odds for Validation Set
    if 'MaxH' in df.columns:
        odds_full = df[['MaxH', 'MaxD', 'MaxA']].values
    elif 'OddsHome' in df.columns:
        odds_full = df[['OddsHome', 'OddsDraw', 'OddsAway']].values
    else:
        odds_full = df[['B365H', 'B365D', 'B365A']].values
        
    odds_val = odds_full[train_end:val_end]
    
    # 3. Load Models & Predict
    cb_pipeline = load_catboost_pipeline(models_dir)
    poisson_params = load_poisson(models_dir)
    
    with open(models_dir / 'metadata_v1_latest.json') as f:
        cb_cols = json.load(f)['features']
        
    logger.info("Generating predictions...")
    p_cb = get_pipeline_probs(cb_pipeline, val_df, cb_cols)
    p_ps = get_poisson_probs(poisson_params, val_df)
    
    # 4. Optimization Loop
    leagues = val_df['League'].unique()
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {"default": {"policy": "SKIP", "weights": {"catboost":0,"neural":0,"poisson":0}}, "leagues": {}}
        
    router = LeagueRouter(config_path) # to get slugs
    
    logger.info("--- STARTING PRECISION OPTIMIZATION (Step=0.1%) ---")
    
    for league in leagues:
        mask = (val_df['League'] == league).values
        if mask.sum() < 20: continue
            
        p_cb_l = p_cb[mask]
        p_ps_l = p_ps[mask]
        y_l = y_val[mask]
        odds_l = odds_val[mask]
        
        # Optimize for ROI directly
        w_c, w_p, best_roi = optimize_weights_1d(y_l, p_cb_l, p_ps_l, odds_l, step=0.001)
        
        logger.info(f"{league.upper():<12}: Best Weights [CB={w_c:.3f}, PS={w_p:.3f}] | Best ROI={best_roi:.2f}% ({len(y_l)} matches)")
        
        slug = router.league_map.get(league.lower(), league.lower())
        
        if slug not in config['leagues']:
            config['leagues'][slug] = {
                "policy": "ENSEMBLE", 
                "description": f"Auto-Optimized ROI={best_roi:.2f}%", 
                "weights": {}
            }
            
        config['leagues'][slug]['weights'] = {
            "catboost": round(float(w_c), 3),
            "neural": 0.0,
            "poisson": round(float(w_p), 3)
        }
        
    # Global Fallback
    logger.info("--- GLOBAL FALLBACK ---")
    w_c_g, w_p_g, roi_g = optimize_weights_1d(y_val, p_cb, p_ps, odds_val, step=0.001)
    logger.info(f"Global Best: CB={w_c_g:.3f}, PS={w_p_g:.3f} | ROI={roi_g:.2f}%")
    
    config['default']['weights'] = {
        "catboost": round(float(w_c_g), 3), 
        "neural": 0.0, 
        "poisson": round(float(w_p_g), 3)
    }
        
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    logger.info(f"Updated {config_path} with precision weights.")

if __name__ == "__main__":
    main()
