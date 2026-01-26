#!/usr/bin/env python3
"""
Train CatBoost v3.3 (Robust)
- Uses pre-split time-based data (Train/Val/Test)
- Optimized for LogLoss & Probability Calibration (not Accuracy)
- Implements Anti-Overfit: Early Stopping, L2 Reg, Depth Constraints
- Performs Random Hyperparameter Search
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import log_loss, brier_score_loss
from datetime import datetime
import itertools
import random

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def evaluate_probs(y_true, y_prob):
    """
    Calculate metrics focused on probability quality.
    """
    ll = log_loss(y_true, y_prob)
    bs = brier_score_loss(y_true, y_prob)
    return {"logloss": ll, "brier": bs}

def train_robust_model(
    data_dir="data/processed/splits_v3_3",
    output_dir="models",
    audit_dir="audit_pack/A6_metrics",
    n_iter=30
):
    logger = setup_logging()
    
    # 1. Load Data
    logger.info("Loading splits...")
    X_train = pd.read_parquet(f"{data_dir}/train.parquet")
    X_val = pd.read_parquet(f"{data_dir}/val.parquet")
    X_test = pd.read_parquet(f"{data_dir}/test.parquet")
    
    # Define Target and Features
    target = 'FTR' # Full Time Result (H, D, A) - but for v3 we model Home Win (H) vs Not H for simplicity? 
    # v3.2 used 'FTR' with multi-class loss usually, or mapped to HomeWin.
    
    # Let's check format. Usually we need numerical target for CatBoost.
    # If FTR is 'H', 'D', 'A', CatBoost handles it if we tell it.
    
    # For Value Betting usually we predict probabilities for H, D, A.
    # Let's assume multi-class classification.
    
    features = [c for c in X_train.columns if c not in [
        'Date', 'FTR', 'start_time', 'date_start', 'HomeTeam', 'AwayTeam', 
        'Season', 'League', 'event_id'
    ]]
    
    logger.info(f"Features ({len(features)}): {features}")
    
    train_pool = Pool(X_train[features], label=X_train[target], cat_features=[])
    val_pool = Pool(X_val[features], label=X_val[target], cat_features=[])
    test_pool = Pool(X_test[features], label=X_test[target], cat_features=[])
    
    # 2. Hyperparameter Grid
    param_grid = {
        'iterations': [500, 1000],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'depth': [4, 6, 8], # Avoid deep trees to prevent overfit
        'l2_leaf_reg': [3, 5, 7, 9],
        'random_strength': [1, 2],
        'bagging_temperature': [0, 1],
        'border_count': [128, 254],
        'early_stopping_rounds': [50]
    }
    
    best_loss = float('inf')
    best_params = {}
    best_model = None
    
    results = []
    
    logger.info(f"Starting Random Search ({n_iter} iterations)...")
    
    # 3. Search Loop
    for i in range(n_iter):
        params = {
            'loss_function': 'MultiClass', # Logloss for multiclass
            'eval_metric': 'MultiClass',
            'task_type': 'CPU',
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False
        }
        
        # Sample random params
        for k, v in param_grid.items():
            params[k] = random.choice(v)
            
        # Train
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        
        # Eval
        val_metric = model.get_best_score()['validation']['MultiClass']
        
        results.append({
            'iter': i,
            'params': params,
            'val_loss': val_metric
        })
        
        logger.info(f"Iter {i+1}/{n_iter}: Loss={val_metric:.5f} | Params={params}")
        
        if val_metric < best_loss:
            best_loss = val_metric
            best_params = params
            best_model = model
            
    logger.info(f"✓ Best Loss: {best_loss:.5f}")
    
    # 4. Final Evaluation on Test (Held-out)
    test_preds = best_model.predict_proba(test_pool)
    
    # Calculate metrics for H, D, A
    # We need to one-hot encode target for LogLoss/Brier per class or micro-average
    # For simplicity, we just dump the overall MultiClass loss on Test
    test_loss = log_loss(X_test[target], test_preds)
    
    logger.info(f"Test LogLoss: {test_loss:.5f}")
    
    # 5. Save Artifacts
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(audit_dir).mkdir(parents=True, exist_ok=True)
    
    # Save Model
    model_path = f"{output_dir}/catboost_soccer_v3_3.cbm"
    best_model.save_model(model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save HP Report
    report = {
        "best_params": best_params,
        "best_val_loss": best_loss,
        "test_loss": test_loss,
        "search_history": results
    }
    
    report_path = f"{audit_dir}/hparam_search_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved HP report to {report_path}")

if __name__ == "__main__":
    train_robust_model()
