#!/usr/bin/env python3
"""
Meta-Stacker Training Script (The "Brain" of the Holy Trinity)

Goal: Train a Meta-Model to combine predictions from:
1. CatBoost (Fundamentals)
2. Neural Network (Market + Sentiment + Fundamentals)
3. Poisson (Statistical Baseline)

Critically, this Meta-Model also sees "Context Features" (Volatility, Divergence)
to decide WHICH model to trust in specific situations.

Method: Stacking
1. Base Models are trained on TRAIN set.
2. Meta Model is trained on VALIDATION set (using Base Model predictions as features).
3. Final Evaluation is on TEST set.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import logging
import torch
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import log_loss, accuracy_score
from scipy.stats import poisson

# Import Neural Network Classes
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.train_neural_model import DenseNN, NeuralModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Loaders (Reused from optimize weights) ---

def load_catboost(models_dir):
    model_path = models_dir / 'catboost_v1_latest.pkl'
    calib_path = models_dir / 'calibrator_v1_latest.pkl'
    
    logger.info(f"Loading CatBoost Base: {model_path}")
    model = joblib.load(model_path)
    
    calibrator = None
    if calib_path.exists():
        logger.info(f"Loading Calibrator: {calib_path}")
        calibrator = joblib.load(calib_path)
        
    return model, calibrator

def load_poisson(models_dir):
    path = models_dir / 'poisson_v1_latest.pkl'
    logger.info(f"Loading Poisson Base: {path}")
    with open(path, 'rb') as f:
        params = joblib.load(f)
    return params

def load_neural(models_dir):
    # Load metadata to get input dim
    meta_path = models_dir / 'neural_metadata_v1_latest.json'
    # Fallback search
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
        
    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)
    
    # Handle state dict vs full checkpoint
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            scaler = checkpoint.get('scaler') # Try to load scaler!
        else:
            model.load_state_dict(checkpoint)
            scaler = None
    else:
        # Very old pytorch format?
        model.load_state_dict(checkpoint)
        scaler = None
            
    model.eval()
    return model, scaler

# --- Predictors ---

def get_catboost_probs(model, calibrator, df, feature_names):
    # Ensure cols
    X = df[feature_names].copy()
    for c in feature_names:
        if c not in X.columns: X[c] = 0
    
    if calibrator:
        return calibrator.predict_proba(X)
    else:
        return model.predict_proba(X)

def get_poisson_probs(params, df):
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

def get_neural_probs(model, scaler, df):
    # Neural needs scaling! 
    # Must match train_neural_model.py logic
    exclude_cols = ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'FTR', 'League',
                    'FTHG', 'FTAG', 'GoalDiff', 'TotalGoals', 'index',
                    'HomeEloAfter', 'AwayEloAfter']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    # 1. Handle NaNs
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    
    # 2. Filter Constant Features (MUST MATCH TRAINING LOGIC)
    # The scaler expects 35 features. We have 39 (inc constant).
    # We must drop the columns that were dropped during training.
    # In train_neural_model.py, it dropped: ['SentimentHome', 'SentimentAway', 'HomeInjury', 'AwayInjury']
    
    cols_to_drop = ['SentimentHome', 'SentimentAway', 'HomeInjury', 'AwayInjury']
    keep_indices = [i for i, c in enumerate(feature_cols) if c not in cols_to_drop]
    
    X = X[:, keep_indices]
    
    # Scale if we have the scaler, otherwise we are in trouble (CRITICAL)
    if scaler:
        X_scaled = scaler.transform(X)
    else:
        logger.warning("No Scaler found in checkpoint! Re-fitting on 70% of data (approx).")
        from sklearn.preprocessing import StandardScaler
        tmp_scaler = StandardScaler()
        n = len(X)
        train_end = int(n * 0.70)
        tmp_scaler.fit(X[:train_end])
        X_scaled = tmp_scaler.transform(X)
    
    trainer = NeuralModelTrainer(model)
    return trainer.predict_proba(X_scaled)

# --- ROI Calc ---
def calculate_ev_roi(probs, odds, y_true):
    ev = (probs * odds) - 1
    staked = 0.0
    profit = 0.0
    bet_mask = ev > 0.05
    
    for i in range(3):
        mask = bet_mask[:, i]
        if mask.sum() == 0: continue
        staked += mask.sum()
        won = (y_true == i)
        pnl = np.where(won[mask], odds[mask, i] - 1, -1.0)
        profit += pnl.sum()
        
    if staked == 0: return 0.0
    return (profit / staked) * 100

# --- MAIN ---

def main():
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / 'data' / 'processed' / 'multi_league_features_2021_2024.csv'
    models_dir = base_dir / 'models'
    
    logger.info("="*60)
    logger.info("META-STACKER TRAINING")
    logger.info("="*60)
    
    # 1. Load Data
    df = pd.read_csv(data_file)
    if 'date' in df.columns: # normalize
         df = df.rename(columns={'date': 'Date', 'home_team': 'HomeTeam', 'away_team': 'AwayTeam', 'odds_1': 'OddsHome'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # 2. Add FTR/Cols if missing (Standard Cleanup)
    if 'FTR' not in df.columns and 'home_goals' in df.columns:
        conditions = [(df['home_goals'] > df['away_goals']), (df['home_goals'] < df['away_goals'])]
        df['FTR'] = np.select(conditions, ['H', 'A'], default='D')
    result_map = {'H': 0, 'D': 1, 'A': 2}
    y = df['FTR'].map(result_map).values
    
    # 3. Load Base Models
    catboost_model, cb_calib = load_catboost(models_dir)
    poisson_params = load_poisson(models_dir)
    neural_model, nn_scaler = load_neural(models_dir)
    
    # Load feature names for CatBoost
    with open(models_dir / 'metadata_v1_latest.json') as f:
        cb_features = json.load(f)['features']
    
    # 4. Generate Predictions (Stacking Features)
    logger.info("Generating Base Model Predictions (This may take a moment)...")
    
    p_cb = get_catboost_probs(catboost_model, cb_calib, df, cb_features)
    p_ps = get_poisson_probs(poisson_params, df)
    p_nn = get_neural_probs(neural_model, nn_scaler, df)
    
    # 5. Build Meta-Dataset
    # Features: Base Probs + Context Features (Volatility, Divergence, Elo)
    
    # Context Features (Must match source DF columns)
    context_features = ['Odds_Volatility', 'Sharp_Divergence', 'Market_Consensus', 'HomeEloBefore', 'EloDiff']
    # Fill defaults if missing in DF (e.g. if script ran on old data)
    for c in context_features:
        if c not in df.columns: df[c] = 0.0
        
    X_context = df[context_features].values
    
    # Combine: [CB_H, CB_D, CB_A, NN_H..., PS_H..., Volatility, Divergence...]
    X_meta = np.hstack([p_cb, p_nn, p_ps, X_context])
    
    feature_names_meta = (
        ['CB_H', 'CB_D', 'CB_A'] + 
        ['NN_H', 'NN_D', 'NN_A'] + 
        ['PS_H', 'PS_D', 'PS_A'] + 
        context_features
    )
    
    logger.info(f"Meta-Model Features: {feature_names_meta}")
    
    # 6. Split Data
    # TRAIN Base models were trained on first 70%
    # We must train META model on VALIDATION (70-85%)
    # And Test on TEST (85-100%)
    
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    # X_train_meta = X_meta[train_end:val_end] # Train on Val set!
    # y_train_meta = y[train_end:val_end]
    
    # Actually, to make it robust, we should combine Train+Val for the Meta-Model 
    # IF we had cross-validated predictions. But we don't. We have predictions from models trained on 'train_end'.
    # So predicting on 'train_end' (first 70%) would be "predicting on training data" (Leakage).
    # So we MUST train the Meta-Learner ONLY on the Validation slice.
    
    X_train_meta = X_meta[train_end:val_end]
    y_train_meta = y[train_end:val_end]
    
    X_test_meta = X_meta[val_end:]
    y_test_meta = y[val_end:]
    odds_test = df[['OddsHome', 'OddsDraw', 'OddsAway']].values[val_end:]
    
    logger.info(f"Meta-Training Samples (Val Set): {len(X_train_meta)}")
    logger.info(f"Meta-Test Samples (Test Set): {len(X_test_meta)}")
    
    # 7. Train Meta-Model (CatBoost Classifier)
    # Using a small tree depth because input is already high-level probabilities
    meta_model = CatBoostClassifier(
        iterations=500,
        depth=4, 
        learning_rate=0.05,
        loss_function='MultiClass',
        verbose=100,
        allow_writing_files=False,
        random_seed=42
    )
    
    meta_model.fit(X_train_meta, y_train_meta, verbose=False)
    
    # 8. Evaluate
    p_test_meta = meta_model.predict_proba(X_test_meta)
    
    loss = log_loss(y_test_meta, p_test_meta)
    acc = accuracy_score(y_test_meta, np.argmax(p_test_meta, axis=1))
    roi = calculate_ev_roi(p_test_meta, odds_test, y_test_meta)
    
    logger.info(f"\nRESULTS (Meta-Stacker on Test Set):")
    logger.info(f"  Log Loss: {loss:.4f}")
    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info(f"  ROI: {roi:.2f}%")
    
    # Feature Importance
    logger.info("\nMeta-Feature Importance:")
    for score, name in sorted(zip(meta_model.get_feature_importance(), feature_names_meta), reverse=True):
        logger.info(f"  {name}: {score:.2f}")
        
    # 9. Save
    meta_model_path = models_dir / 'meta_stacker_v1.cbm'
    meta_model.save_model(str(meta_model_path))
    
    # Save Feature Names for Inference
    with open(models_dir / 'meta_stacker_metadata.json', 'w') as f:
        json.dump({
            'feature_names': feature_names_meta,
            'metrics': {'log_loss': loss, 'roi': roi},
            'timestamp': str(pd.Timestamp.now())
        }, f, indent=2)
        
    logger.info(f"Saved Meta-Model to {meta_model_path}")

if __name__ == "__main__":
    main()
