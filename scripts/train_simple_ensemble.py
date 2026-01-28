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
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    data_file = base_dir / 'data' / 'processed' / 'epl_historical_2021_2024.csv'
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Split
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Load models
    logger.info("\nLoading models...")
    poisson = PoissonMatchPredictor.load(base_dir / 'models' / 'poisson_v1_latest.pkl')
    catboost = ModelLoader()
    catboost.load_latest()
    
    # Load CatBoost features for predictions
    features_df = pd.read_csv(base_dir / 'data' / 'processed' / 'epl_features_2021_2024.csv')
    features_df['Date'] = pd.to_datetime(features_df['Date'])
    
    result_map = {'H': 0, 'D': 1, 'A': 2}
    
    # VALIDATION SET (for calibration)
    logger.info("\nValidation set...")
    
    # Filter CatBoost features to exact same dates and sort
    val_dates = val_df['Date'].values
    val_features = features_df[features_df['Date'].isin(val_dates)].copy()
    val_features = val_features.sort_values('Date').reset_index(drop=True)
    
    # Also sort val_df
    val_df_sorted = val_df.sort_values('Date').reset_index(drop=True)
    
    # Double-check lengths match
    if len(val_features) != len(val_df_sorted):
        logger.warning(f"Mismatch: {len(val_df_sorted)} hist vs {len(val_features)} features")
        # Find common dates
        common_dates = set(val_df_sorted['Date']) & set(val_features['Date'])
        logger.info(f"Using {len(common_dates)} common dates")
        val_df_sorted = val_df_sorted[val_df_sorted['Date'].isin(common_dates)].reset_index(drop=True)
        val_features = val_features[val_features['Date'].isin(common_dates)].reset_index(drop=True)
    
    feature_cols = [col for col in val_features.columns 
                    if col not in ['Date', 'Season', 'FTR', 'HomeTeam', 'AwayTeam']]
    
    # Use DataFrame directly to preserve column names for ColumnTransformer
    X_val = val_features[feature_cols]
    
    # Filter to CatBoost's expected features (22 features)
    expected_features = catboost.get_feature_names()
    if expected_features and len(expected_features) > 0:
        # Keep only features CatBoost expects
        missing = [f for f in expected_features if f not in X_val.columns]
        extra = [f for f in X_val.columns if f not in expected_features]
        if missing:
            logger.warning(f"Missing features: {missing[:5]}...")
            for f in missing:
                X_val[f] = 0.0
        if extra:
            logger.info(f"Dropping extra features: {len(extra)} cols")
        X_val = X_val[expected_features]
    
    logger.info(f"Aligned: {len(val_df_sorted)} matches, {X_val.shape[1]} features")
    
    # Get labels
    y_val = val_df_sorted['FTR'].map(result_map).values
    
    # Predictions
    poisson_val = poisson.predict(val_df_sorted)[['prob_home', 'prob_draw', 'prob_away']].values
    catboost_val = catboost.predict(X_val)
    
    # Ensure same length (slice to minimum)
    min_len = min(len(poisson_val), len(catboost_val))
    if len(poisson_val) != len(catboost_val):
        logger.warning(f"Length mismatch: Poisson={len(poisson_val)}, CatBoost={len(catboost_val)}")
        logger.info(f"Using first {min_len} predictions")
        poisson_val = poisson_val[:min_len]
        catboost_val = catboost_val[:min_len]
        y_val_sliced = y_val[:min_len]
    else:
        y_val_sliced = y_val
    
    # Average (simple ensemble)
    ensemble_val = (poisson_val + catboost_val) / 2.0
    
    # Calibrate
    logger.info("Calibrating ensemble...")
    calibrators = calibrate_ensemble(ensemble_val, y_val_sliced)
    
    # TEST SET
    logger.info("\nTest set...")
    test_dates = test_df['Date'].values
    test_features = features_df[features_df['Date'].isin(test_dates)].copy()
    test_features = test_features.sort_values('Date').reset_index(drop=True)
    
    test_df_sorted = test_df.sort_values('Date').reset_index(drop=True)
    
    # Ensure alignment
    if len(test_features) != len(test_df_sorted):
        common_dates = set(test_df_sorted['Date']) & set(test_features['Date'])
        logger.info(f"Using {len(common_dates)} common test dates")
        test_df_sorted = test_df_sorted[test_df_sorted['Date'].isin(common_dates)].reset_index(drop=True)
        test_features = test_features[test_features['Date'].isin(common_dates)].reset_index(drop=True)
    
    X_test = test_features[feature_cols]
    
    # Filter to CatBoost's expected features
    if expected_features and len(expected_features) > 0:
        missing = [f for f in expected_features if f not in X_test.columns]
        extra = [f for f in X_test.columns if f not in expected_features]
        if missing:
            for f in missing:
                X_test[f] = 0.0
        X_test = X_test[expected_features]
    
    logger.info(f"Test aligned: {len(test_df_sorted)} matches, {X_test.shape[1]} features")
    
    # Get labels
    y_test = test_df_sorted['FTR'].map(result_map).values
    
    poisson_test = poisson.predict(test_df_sorted)[['prob_home', 'prob_draw', 'prob_away']].values
    catboost_test = catboost.predict(X_test)
    
    # Ensure same length
    min_len_test = min(len(poisson_test), len(catboost_test))
    if len(poisson_test) != len(catboost_test):
        logger.warning(f"Test length mismatch: Poisson={len(poisson_test)}, CatBoost={len(catboost_test)}")
        logger.info(f"Using first {min_len_test} predictions")
        poisson_test = poisson_test[:min_len_test]
        catboost_test = catboost_test[:min_len_test]
        y_test = y_test[:min_len_test]
    
    # Average
    ensemble_test_raw = (poisson_test + catboost_test) / 2.0
    
    # Calibrate
    ensemble_test = apply_calibration(ensemble_test_raw, calibrators)
    
    y_test = test_df_sorted['FTR'].map(result_map).values
    
    # EVALUATION
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
        logloss = log_loss(y_test, probs)
        
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
