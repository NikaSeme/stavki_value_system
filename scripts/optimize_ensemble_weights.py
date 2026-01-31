#!/usr/bin/env python3
"""
Optimize Ensemble Weights for Maximum Profit/Accuracy.

Method:
1. Loads 3 pre-trained models (CatBoost, Neural, Poisson).
2. Generates predictions on Validation Set.
3. Performs Grid Search / Optimization to find weights (w_c, w_n, w_p) that minimize Log Loss.
4. Calculates theoretical profit (ROI) on Test Set using these weights.
5. Saves optimal weights to 'models/ensemble_optimized_metadata.json'.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import logging
from datetime import datetime, timezone
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from scipy.stats import poisson

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.neural_predictor import NeuralPredictor
# We need the Neural classes to load the pytorch model directly if cleaner
from scripts.train_neural_model import DenseNN, NeuralModelTrainer 
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Loaders ---

def load_catboost(models_dir):
    model_path = models_dir / 'catboost_v1_latest.pkl'
    calib_path = models_dir / 'calibrator_v1_latest.pkl'
    
    logger.info(f"Loading CatBoost: {model_path}")
    model = joblib.load(model_path)
    
    calibrator = None
    if calib_path.exists():
        logger.info(f"Loading Calibrator: {calib_path}")
        calibrator = joblib.load(calib_path)
        
    return model, calibrator

def load_poisson(models_dir):
    path = models_dir / 'poisson_v1_latest.pkl'
    logger.info(f"Loading Poisson: {path}")
    with open(path, 'rb') as f:
        params = joblib.load(f)
    return params # dict

def load_neural(models_dir):
    # Load metadata to get input dim
    meta_path = models_dir / 'neural_metadata_v1_latest.json'
    # Fallback search if symlink missing
    if not meta_path.exists():
         meta_path = sorted(models_dir.glob('neural_metadata_*.json'))[-1]
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    input_dim = meta['architecture']['input_dim']
    model = DenseNN(input_dim=input_dim)
    
    # Load weights
    pt_path = models_dir / 'neural_v1_latest.pt'
    if not pt_path.exists():
        pt_path = sorted(models_dir.glob('neural_v1_*.pt'))[-1]
        
    try:
        checkpoint = torch.load(pt_path, weights_only=False)
        # Check if it's a full checkpoint or just state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            logger.info("  Loading from full checkpoint dict...")
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
    except TypeError:
        # Fallback for older pytorch/simple load
        checkpoint = torch.load(pt_path)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
    model.eval()
    return model

# --- Predictors ---

def get_catboost_probs(model, calibrator, df, feature_names):
    # Ensure cols
    X = df[feature_names].copy()
    # Fill N/A in case
    for c in feature_names:
        if c not in X.columns: X[c] = 0
    
    if calibrator:
        return calibrator.predict_proba(X)
    else:
        return model.predict_proba(X)

def get_poisson_probs(params, df):
    # Vectorized or loop? Loop is safer for robust code
    home_adv = params['home_advantage']
    att = params['team_attack']
    defe = params['team_defense']
    avg_goals = params['league_avg_goals']
    
    probs = []
    max_goals = 6
    
    for _, row in df.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        h_att = att.get(ht, 1.0)
        h_def = defe.get(ht, 1.0)
        a_att = att.get(at, 1.0)
        a_def = defe.get(at, 1.0)
        
        lam_h = avg_goals * h_att * a_def * (1 + home_adv)
        lam_a = avg_goals * a_att * h_def
        
        p_h, p_d, p_a = 0, 0, 0
        for i in range(max_goals):
            for j in range(max_goals):
                p = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
                if i > j: p_h += p
                elif i == j: p_d += p
                else: p_a += p
        probs.append([p_h, p_d, p_a])
        
    return np.array(probs)

def get_neural_probs(model, df):
    # Neural needs scaling! 
    # We should load the scaler used during training?
    # Or refit scaler on this data? Using same split logic guarantees same distribution.
    # Ideally load scaler artifact.
    # If not saved separately, we might have issue.
    # scripts/train_neural_model.py does NOT save scaler separately?
    # It creates it internally.
    # Wait, checking train_neural_model.py... "trainer.scaler = scaler".
    # But does it save it? No.
    # FIX: Recalculate scaler. Since we have full dataset, we can recreate the split and fit scaler on train.
    
    # Match the logic in train_neural_model.py EXACTLY
    # Exclude: Date, Teams, LEAGUE, Season, FTR (Target), Match Outcomes (Leakage), indices
    # Include: Odds, Form, Points
    # Exclude: Date, Teams, LEAGUE, Season, FTR (Target), Match Outcomes (Leakage), indices
    exclude_cols = ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'FTR', 'League',
                    'FTHG', 'FTAG', 'GoalDiff', 'TotalGoals', 'index',
                    'HomeEloAfter', 'AwayEloAfter']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Filter numeric only for safety
    feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    X = df[feature_cols].values
    
    # Clean
    X = np.nan_to_num(X, nan=0.0)
    std = np.std(X, axis=0)
    X = X[:, std > 1e-6] # Drop constant
    
    # Scale
    scaler = StandardScaler()
    # We must fit only on TRAIN part of this DF
    n = len(X)
    train_end = int(n * 0.70)
    scaler.fit(X[:train_end])
    X_scaled = scaler.transform(X)
    
    # Predict
    model_trainer = NeuralModelTrainer(model) # Helper to use predict_proba
    return model_trainer.predict_proba(X_scaled) 

# --- Optimization ---

def calculate_ev_roi(probs, odds, y_true):
    # Simple strategy: Bet if EV > 0
    # Returns ROI %
    
    # Probs: (N, 3)
    # Odds: (N, 3) (B365H, D, A)
    # y_true: (N,) (0=H, 1=D, 2=A) mapping depends... usually H=0
    
    ev = (probs * odds) - 1
    
    staked = 0.0
    profit = 0.0
    
    # Vectorized
    bet_mask = ev > 0.05 # 5% threshold
    
    for i in range(3): # H, D, A
        mask = bet_mask[:, i]
        if mask.sum() == 0: continue
        
        staked += mask.sum()
        
        # Outcome mask
        won = (y_true == i)
        
        # Profit = (Odds - 1) * 1 for winners - 1 for losers
        # or Returns = Odds * 1 for winners. Profit = Returns - Stake.
        
        pnl = np.where(won[mask], odds[mask, i] - 1, -1.0)
        profit += pnl.sum()
        
    if staked == 0: return 0.0
    return (profit / staked) * 100

def main():
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / 'data' / 'processed' / 'multi_league_features_2021_2024.csv'
    models_dir = base_dir / 'models'
    
    df = pd.read_csv(data_file)
    # Normalize columns if needed
    if 'date' in df.columns:
        feature_map = {
            'date': 'Date', 'home_team': 'HomeTeam', 'away_team': 'AwayTeam', 
            'league': 'League',
            'odds_1': 'OddsHome', 'odds_x': 'OddsDraw', 'odds_2': 'OddsAway'
        }
        df = df.rename(columns=feature_map)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Calculate FTR if missing (build_features drops it)
    if 'FTR' not in df.columns and 'home_goals' in df.columns:
        conditions = [
            (df['home_goals'] > df['away_goals']),
            (df['home_goals'] < df['away_goals'])
        ]
        choices = ['H', 'A']
        df['FTR'] = np.select(conditions, choices, default='D')

    # Mapping
    result_map = {'H': 0, 'D': 1, 'A': 2}
    y = df['FTR'].map(result_map).values
    
    # Get Odds
    # Map B365 or OddsHome
    if 'OddsHome' in df.columns:
        odds = df[['OddsHome', 'OddsDraw', 'OddsAway']].values
    else:
        odds = df[['B365H', 'B365D', 'B365A']].values
        
    # Load Models
    catboost_model, calibrator = load_catboost(models_dir)
    poisson_params = load_poisson(models_dir)
    # Neural... complex due to scaler. For now, check if we can skip re-scaling if weights match?
    # To do it properly, we need to replicate train_neural_model logic.
    # For MVP optimization, let's assume Neural works or re-fit scaler on train section.
    
    # features
    with open(models_dir / 'metadata_v1_latest.json') as f:
        cb_features = json.load(f)['features']
        
    # Generate Predictions (All Matches)
    logger.info("Generating Model Predictions...")
    
    # CatBoost
    p_cb = get_catboost_probs(catboost_model, calibrator, df, cb_features)
    
    # Poisson
    p_ps = get_poisson_probs(poisson_params, df)
    
    # Neural
    try:
        neural_model = load_neural(models_dir)
        p_nn = get_neural_probs(neural_model, df)
    except Exception as e:
        logger.error(f"Neural unavailable for optimization: {e}")
        p_nn = np.zeros_like(p_cb) # Fallback 0
        
    # Split Data (Validation Set is the target for optimization)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    # We optimize on Validation, Test on Test
    
    # Helper to slice
    def get_slice(arr, start, end):
        return arr[start:end]
        
    p_cb_val = get_slice(p_cb, train_end, val_end)
    p_nn_val = get_slice(p_nn, train_end, val_end)
    p_ps_val = get_slice(p_ps, train_end, val_end)
    
    y_val_slice = y[train_end:val_end]
    odds_val = odds[train_end:val_end]
    
    # Optimization Loop
    logger.info("Optimizing Weights...")
    
    best_loss = float('inf')
    best_weights = (0.5, 0.3, 0.2)
    
    # Grid Search (Coarse is safely convex usually)
    steps = np.arange(0, 1.05, 0.05)
    
    results = []
    
    # Neural Network Enabled!
    # Search grid for w_n
    # To keep it simple O(N^2), let's loop w_n too?
    # Or just implied? No, we need 2 loops for 3 weights.
        
    # Re-writing the loop for 3-model grid search
    # Neural Network DISABLED per user request (maximize 2-model ensemble)
    logger.info("Using 2-Model Ensemble (CatBoost + Poisson) - Neural Network Disabled")
    
    for w_c in steps:
        # Force Neural to 0.0
        w_n = 0.0
        
        # w_p is remaining
        if w_c > 1.0: continue
        w_p = 1.0 - w_c
        
        # Combine
        p_ensemble = w_c * p_cb_val + w_n * p_nn_val + w_p * p_ps_val
        
        # Norm (avoid tiny errors)
        row_sums = p_ensemble.sum(axis=1, keepdims=True)
        p_ensemble /= row_sums
        
        loss = log_loss(y_val_slice, p_ensemble)
        
        results.append((loss, w_c, w_n, w_p))
        
        if loss < best_loss:
            best_loss = loss
            best_weights = (w_c, w_n, w_p)
        

                
    logger.info(f"Optimal Weights (Validation): CB={best_weights[0]:.2f}, NN={best_weights[1]:.2f}, PS={best_weights[2]:.2f}")
    logger.info(f"Best Validation LogLoss: {best_loss:.4f}")
    
    # Test on Test Set
    p_cb_test = get_slice(p_cb, val_end, n)
    p_nn_test = get_slice(p_nn, val_end, n)
    p_ps_test = get_slice(p_ps, val_end, n)
    y_test_slice = y[val_end:n]
    odds_test = odds[val_end:n]
    
    w_c, w_n, w_p = best_weights
    p_test = w_c * p_cb_test + w_n * p_nn_test + w_p * p_ps_test
    p_test /= p_test.sum(axis=1, keepdims=True)
    
    test_loss = log_loss(y_test_slice, p_test)
    roi = calculate_ev_roi(p_test, odds_test, y_test_slice)
    
    logger.info(f"TEST RESULTS:")
    logger.info(f"  Log Loss: {test_loss:.4f}")
    logger.info(f"  Est. Model ROI: {roi:.2f}% (Theoretical on >5% EV bets)")
    
    # Save
    output_meta = models_dir / 'ensemble_optimized_metadata.json'
    with open(output_meta, 'w') as f:
        json.dump({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'weights': {
                'catboost': best_weights[0],
                'neural': best_weights[1],
                'poisson': best_weights[2]
            },
            'metrics': {
                'val_log_loss': best_loss,
                'test_log_loss': test_loss,
                'test_roi': roi
            }
        }, f, indent=2)
        
    logger.info(f"Saved optimized weights to {output_meta}")

if __name__ == '__main__':
    main()
