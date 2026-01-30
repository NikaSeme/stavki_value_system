"""
Simple ensemble: average Poisson + CatBoost probabilities.

Simpler than stacking, but still effective by combining diverse models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import sys
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ModelLoader
from scripts.train_poisson_model import PoissonMatchPredictor
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sklearn.isotonic import IsotonicRegression
from src.logging_setup import get_logger
from catboost import Pool

logger = get_logger(__name__)


def calibrate_ensemble(probs_train, y_train, method='platt'):
    """
    Apply calibration to ensemble probabilities.
    
    Args:
        probs_train: Raw probabilities from ensemble (N x 3)
        y_train: True labels (N,)
        method: 'platt' (sigmoid) or 'isotonic' (default: platt)
    
    Returns:
        List of calibrators (one per class)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    
    calibrators = []
    
    for class_idx in range(3):
        y_binary = (y_train == class_idx).astype(int)
        
        if method == 'platt':
            # Platt Scaling: Fit sigmoid (logistic regression)
            # Smooth extrapolation for out-of-distribution data
            calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
            X = probs_train[:, class_idx].reshape(-1, 1)
            calibrator.fit(X, y_binary)
        else:
            # Isotonic Regression (legacy)
            # WARNING: Poor extrapolation on unseen leagues!
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(probs_train[:, class_idx], y_binary)
        
        calibrators.append(calibrator)
    
    return calibrators


def apply_calibration(probs, calibrators):
    """
    Apply calibration and renormalize.
    
    Supports both Platt (LogisticRegression) and Isotonic calibrators.
    """
    calibrated = np.zeros_like(probs)
    
    for i, cal in enumerate(calibrators):
        # Check calibrator type
        if hasattr(cal, 'predict_proba'):
            # Platt scaling (LogisticRegression)
            X = probs[:, i].reshape(-1, 1)
            calibrated[:, i] = cal.predict_proba(X)[:, 1]
        else:
            # Isotonic regression (legacy)
            calibrated[:, i] = cal.predict(probs[:, i])
    
    # Renormalize to sum to 1.0
    row_sums = calibrated.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums = np.maximum(row_sums, 1e-10)
    calibrated = calibrated / row_sums
    
    return calibrated


def main():
    """Train simple ensemble."""
    logger.info("=" * 70)
    logger.info("SIMPLE ENSEMBLE (AVERAGING)")
    logger.info("=" * 70)
    
    base_dir = Path(__file__).parent.parent
    
    # Load data
    data_file = base_dir / 'data' / 'processed' / 'multi_league_features_2021_2024.csv'
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Split: 60% train, 15% calibration, 10% validation, 15% test
    n = len(df)
    train_end = int(n * 0.60)
    cal_end = int(n * 0.75)    # NEW: Calibration set
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    cal_df = df.iloc[train_end:cal_end].copy().reset_index(drop=True)
    val_df = df.iloc[cal_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    
    logger.info(f"Train: {len(train_df)}, Cal: {len(cal_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Load models
    logger.info("\nLoading models...")
    poisson = PoissonMatchPredictor.load(base_dir / 'models' / 'poisson_v1_latest.pkl')
    catboost = ModelLoader()
    catboost.load_latest()
    
    result_map = {'H': 0, 'D': 1, 'A': 2}
    
    # --- CALIBRATION SET ---
    logger.info("\nCalibration set...")
    
    # 1. Select Features
    # Exclude leakage and meta columns
    exclude_cols = ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'FTR', 'League',
                    'FTHG', 'FTAG', 'GoalDiff', 'TotalGoals']
    
    feature_cols = [col for col in cal_df.columns if col not in exclude_cols]
    
    # 2. Build Raw DataFrame for Scaler
    X_cal_raw = cal_df[feature_cols].copy()
    
    # Add Categoricals (needed for passthrough in ColumnTransformer)
    # Ensure no NaN and forced string type
    X_cal_raw['HomeTeam'] = cal_df['HomeTeam'].fillna("Unknown").astype(str)
    X_cal_raw['AwayTeam'] = cal_df['AwayTeam'].fillna("Unknown").astype(str)
    X_cal_raw['League'] = cal_df['League'].fillna('unknown').astype(str)
    
    # Handle NaN in numerics
    num_cols = [c for c in feature_cols if c not in ['HomeTeam', 'AwayTeam', 'League']]
    X_cal_raw[num_cols] = X_cal_raw[num_cols].fillna(0.0)

    # 3. Standardize using loaded scaler
    try:
        X_cal_transformed = catboost.scaler.transform(X_cal_raw)
    except Exception as e:
        logger.error(f"Scaler failed: {e}. Check feature columns match!")
        raise e
        
    # 4. CatBoost Prediction
    # CatBoost Pool expects indices for categoricals (last 3 columns)
    n_num = X_cal_transformed.shape[1] - 3
    cat_indices = [n_num, n_num+1, n_num+2]
    
    cal_pool = Pool(X_cal_transformed, cat_features=cat_indices)
    catboost_cal = catboost.model.predict_proba(cal_pool)
    
    # 5. Poisson Prediction
    poisson_cal = poisson.predict(cal_df)[['prob_home', 'prob_draw', 'prob_away']].values
    
    # 6. Ensemble Average
    ensemble_cal = (poisson_cal + catboost_cal) / 2.0
    
    # 7. Calibrate
    y_cal = cal_df['FTR'].map(result_map).values
    logger.info("Calibrating ensemble on calibration set...")
    calibrators = calibrate_ensemble(ensemble_cal, y_cal)
    
    
    # --- TEST SET ---
    logger.info("\nTest set...")
    
    # 1. Build Raw DataFrame
    X_test_raw = test_df[feature_cols].copy()
    X_test_raw['HomeTeam'] = test_df['HomeTeam'].fillna("Unknown").astype(str)
    X_test_raw['AwayTeam'] = test_df['AwayTeam'].fillna("Unknown").astype(str)
    X_test_raw['League'] = test_df['League'].fillna('unknown').astype(str)
    X_test_raw[num_cols] = X_test_raw[num_cols].fillna(0.0)
    
    # 2. Transform/Predict
    X_test_transformed = catboost.scaler.transform(X_test_raw)
    test_pool = Pool(X_test_transformed, cat_features=cat_indices)
    catboost_test = catboost.model.predict_proba(test_pool)
    
    # 3. Poisson Predict
    poisson_test = poisson.predict(test_df)[['prob_home', 'prob_draw', 'prob_away']].values
    
    # 4. Average & Calibrate
    ensemble_test_raw = (poisson_test + catboost_test) / 2.0
    ensemble_test = apply_calibration(ensemble_test_raw, calibrators)
    
    y_test = test_df['FTR'].map(result_map).values
    
    
    # --- EVALUATION ---
    logger.info("\n" + "=" * 70)
    logger.info("TEST SET RESULTS")
    logger.info("=" * 70)
    
    # Calculate metrics for all three
    models = {
        'Poisson (A)': poisson_test,
        'CatBoost (B)': catboost_test,
        'Ensemble': ensemble_test
    }
    
    results = {}
    
    for name, probs in models.items():
        y_pred = probs.argmax(axis=1)
        acc = accuracy_score(y_test, y_pred)
        
        # Brier
        brier_scores = []
        for i in range(3):
            y_binary = (y_test == i).astype(int)
            brier = brier_score_loss(y_binary, probs[:, i])
            brier_scores.append(brier)
        avg_brier = np.mean(brier_scores)
        
        # Log loss
        try:
            logloss = log_loss(y_test, probs)
        except:
            logloss = 9.99
        
        results[name] = {
            'accuracy': acc,
            'brier': avg_brier,
            'logloss': logloss
        }
    
    # Print table
    logger.info("\n| Metric   | Poisson (A) | CatBoost (B) | Ensemble | Winner |")
    logger.info("|----------|-------------|--------------|----------|--------|")
    
    # Accuracy
    accs = [results[m]['accuracy'] for m in ['Poisson (A)', 'CatBoost (B)', 'Ensemble']]
    best_acc = max(accs)
    winner = ['A', 'B', 'Ens'][accs.index(best_acc)]
    logger.info(f"| Accuracy | {results['Poisson (A)']['accuracy']:11.2%} | {results['CatBoost (B)']['accuracy']:12.2%} | {results['Ensemble']['accuracy']:8.2%} | {winner:6} |")
    
    # Brier
    briers = [results[m]['brier'] for m in ['Poisson (A)', 'CatBoost (B)', 'Ensemble']]
    best_brier = min(briers)
    winner = ['A', 'B', 'Ens'][briers.index(best_brier)]
    logger.info(f"| Brier    | {results['Poisson (A)']['brier']:11.4f} | {results['CatBoost (B)']['brier']:12.4f} | {results['Ensemble']['brier']:8.4f} | {winner:6} |")
    
    # LogLoss
    lls = [results[m]['logloss'] for m in ['Poisson (A)', 'CatBoost (B)', 'Ensemble']]
    best_ll = min(lls)
    winner = ['A', 'B', 'Ens'][lls.index(best_ll)]
    logger.info(f"| LogLoss  | {results['Poisson (A)']['logloss']:11.4f} | {results['CatBoost (B)']['logloss']:12.4f} | {results['Ensemble']['logloss']:8.4f} | {winner:6} |")
    
    # Save ensemble
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = base_dir / 'models'
    
    ensemble_data = {
        'method': 'simple_average',
        'weights': [0.5, 0.5],  # Equal weights
        'models': ['poisson_v1_latest', 'catboost_v1_latest'],
        'calibrators': calibrators,
        'timestamp': timestamp
    }
    
    ensemble_file = model_dir / f'ensemble_simple_{timestamp}.pkl'
    with open(ensemble_file, 'wb') as f:
        pickle.dump(ensemble_data, f)
    
    # Symlink
    ensemble_latest = model_dir / 'ensemble_simple_latest.pkl'
    if ensemble_latest.exists():
        ensemble_latest.unlink()
    ensemble_latest.symlink_to(ensemble_file.name)
    
    # Metadata
    metadata = {
        'model': 'SimpleEnsemble',
        'method': 'average',
        'version': 'v2',  # v2 = Platt scaling
        'train_date': timestamp,
        'base_models': ['Poisson', 'CatBoost'],
        'weights': [0.5, 0.5],
        'calibration': 'Platt',  # Changed from Isotonic
        'calibration_method': 'sigmoid',
        'metrics': {
            'test': {
                'accuracy': float(results['Ensemble']['accuracy']),
                'brier_score': float(results['Ensemble']['brier']),
                'log_loss': float(results['Ensemble']['logloss'])
            }
        }
    }
    
    meta_file = model_dir / f'ensemble_simple_metadata_{timestamp}.json'
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\n✅ Ensemble saved: {ensemble_file}")
    logger.info(f"📊 Metadata: {meta_file}")
    
    # Summary
    improvement = results['CatBoost (B)']['brier'] - results['Ensemble']['brier']
    logger.info(f"\n📈 Brier improvement: {improvement:+.4f}")
    
    if results['Ensemble']['brier'] < min(results['Poisson (A)']['brier'], results['CatBoost (B)']['brier']):
        logger.info("✅ ENSEMBLE BEATS BOTH BASE MODELS!")
    else:
        logger.info("⚠️  Ensemble competitive but not best")


if __name__ == '__main__':
    main()
