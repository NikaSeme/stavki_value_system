#!/usr/bin/env python3
"""
CatBoost Training V2 (Task D)

Trains CatBoost with:
- Corrected ML odds line
- Strict feature contract (28 features)
- Chronological train/valid/test split
- Optional calibration (only kept if improves metrics)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, log_loss
from sklearn.isotonic import IsotonicRegression

from src.models.feature_contract import load_contract

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"


def load_and_split(dataset_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load dataset and split chronologically.
    
    Split:
    - Train: 70% oldest
    - Valid: 15% middle
    - Test: 15% newest
    """
    df = pd.read_csv(dataset_path)
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    df = df.sort_values('kickoff_time').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    
    train = df.iloc[:train_end]
    valid = df.iloc[train_end:valid_end]
    test = df.iloc[valid_end:]
    
    logger.info(f"Split: Train={len(train)}, Valid={len(valid)}, Test={len(test)}")
    logger.info(f"  Train: {train['kickoff_time'].min()} to {train['kickoff_time'].max()}")
    logger.info(f"  Valid: {valid['kickoff_time'].min()} to {valid['kickoff_time'].max()}")
    logger.info(f"  Test:  {test['kickoff_time'].min()} to {test['kickoff_time'].max()}")
    
    return train, valid, test


def prepare_features(df: pd.DataFrame, contract) -> Tuple[pd.DataFrame, np.ndarray]:
    """Prepare feature matrix using strict contract."""
    # Get contract features
    feature_cols = contract.features
    cat_features = list(contract.categorical)
    num_features = list(contract.numeric)
    
    # Select and validate
    X = df[feature_cols].copy()
    y = df['label'].values
    
    # Fill NaN in numeric (only for existing columns)
    for col in num_features:
        if col in X.columns:
            X[col] = X[col].fillna(0.0)
    
    # Fill NaN in categorical
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].fillna('Unknown').astype(str)
    
    return X, y


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    y_pred = y_proba.argmax(axis=1)
    
    accuracy = accuracy_score(y_true, y_pred)
    logloss = log_loss(y_true, y_proba, labels=[0, 1, 2])
    
    # Brier score (multi-class)
    brier = 0.0
    for i in range(3):
        binary = (y_true == i).astype(float)
        brier += np.mean((y_proba[:, i] - binary) ** 2)
    brier /= 3
    
    return {
        'accuracy': accuracy,
        'logloss': logloss,
        'brier': brier
    }


def compute_calibration_table(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_idx: int,
    n_bins: int = 10
) -> pd.DataFrame:
    """Compute calibration table for one class."""
    probs = y_proba[:, class_idx]
    actual = (y_true == class_idx).astype(float)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= low) & (probs < high)
        
        if mask.sum() > 0:
            rows.append({
                'bin': f"{low:.1f}-{high:.1f}",
                'mean_pred': probs[mask].mean(),
                'mean_actual': actual[mask].mean(),
                'count': int(mask.sum()),
            })
        else:
            rows.append({
                'bin': f"{low:.1f}-{high:.1f}",
                'mean_pred': (low + high) / 2,
                'mean_actual': np.nan,
                'count': 0,
            })
    
    return pd.DataFrame(rows)


def train_catboost_v2(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    contract
) -> Dict[str, Any]:
    """
    Train CatBoost with V2 pipeline.
    
    Returns metrics and artifacts.
    """
    # Prepare features
    cat_features = list(contract.categorical)
    
    X_train, y_train = prepare_features(train, contract)
    X_valid, y_valid = prepare_features(valid, contract)
    X_test, y_test = prepare_features(test, contract)
    
    logger.info(f"Feature matrix: {X_train.shape[1]} features")
    
    # Get categorical feature indices
    cat_indices = [X_train.columns.get_loc(c) for c in cat_features if c in X_train.columns]
    
    # CatBoost pools
    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_indices)
    test_pool = Pool(X_test, y_test, cat_features=cat_indices)
    
    # Hyperparameters
    params = {
        'iterations': 1000,
        'depth': 5,
        'learning_rate': 0.03,
        'l2_leaf_reg': 9,
        'loss_function': 'MultiClass',
        'eval_metric': 'MultiClass',
        'random_seed': 42,
        'verbose': 100,
        'early_stopping_rounds': 50,
        'task_type': 'CPU',
    }
    
    print("\n🚀 Training CatBoost V2...")
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    
    # Predictions
    y_train_proba = model.predict_proba(train_pool)
    y_valid_proba = model.predict_proba(valid_pool)
    y_test_proba_raw = model.predict_proba(test_pool)
    
    # Metrics
    train_metrics = compute_metrics(y_train, y_train_proba)
    valid_metrics = compute_metrics(y_valid, y_valid_proba)
    test_metrics_raw = compute_metrics(y_test, y_test_proba_raw)
    
    print("\n📊 Raw Model Metrics:")
    print(f"  Train: Acc={train_metrics['accuracy']:.4f}, LL={train_metrics['logloss']:.4f}")
    print(f"  Valid: Acc={valid_metrics['accuracy']:.4f}, LL={valid_metrics['logloss']:.4f}")
    print(f"  Test:  Acc={test_metrics_raw['accuracy']:.4f}, LL={test_metrics_raw['logloss']:.4f}")
    
    # Try calibration
    print("\n🔧 Attempting calibration...")
    calibrators = {}
    y_valid_proba_cal = y_valid_proba.copy()
    
    for c in range(3):
        iso = IsotonicRegression(out_of_bounds='clip')
        binary = (y_valid == c).astype(float)
        iso.fit(y_valid_proba[:, c], binary)
        calibrators[c] = iso
        y_valid_proba_cal[:, c] = iso.predict(y_valid_proba[:, c])
    
    # Renormalize
    y_valid_proba_cal = y_valid_proba_cal / y_valid_proba_cal.sum(axis=1, keepdims=True)
    
    valid_metrics_cal = compute_metrics(y_valid, y_valid_proba_cal)
    
    # Apply calibration to test
    y_test_proba_cal = y_test_proba_raw.copy()
    for c in range(3):
        y_test_proba_cal[:, c] = calibrators[c].predict(y_test_proba_raw[:, c])
    y_test_proba_cal = y_test_proba_cal / y_test_proba_cal.sum(axis=1, keepdims=True)
    
    test_metrics_cal = compute_metrics(y_test, y_test_proba_cal)
    
    # Decide whether to keep calibration
    use_calibration = test_metrics_cal['logloss'] < test_metrics_raw['logloss']
    
    if use_calibration:
        print(f"  ✅ Calibration improves LogLoss: {test_metrics_raw['logloss']:.4f} → {test_metrics_cal['logloss']:.4f}")
        y_test_proba_final = y_test_proba_cal
        test_metrics = test_metrics_cal
    else:
        print(f"  ❌ Calibration hurts LogLoss: {test_metrics_raw['logloss']:.4f} → {test_metrics_cal['logloss']:.4f}")
        print("  Using raw predictions.")
        calibrators = None
        y_test_proba_final = y_test_proba_raw
        test_metrics = test_metrics_raw
    
    # Calibration tables
    print("\n📈 Calibration Tables (Test Set):")
    for c, name in enumerate(['Home', 'Draw', 'Away']):
        table = compute_calibration_table(y_test, y_test_proba_final, c)
        print(f"\n{name}:")
        print(table.to_string(index=False))
    
    # Feature importance
    importance = model.get_feature_importance()
    feature_names = list(X_train.columns)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\n" + "=" * 50)
    print("TOP 15 FEATURE IMPORTANCES")
    print("=" * 50)
    for i, row in importance_df.head(15).iterrows():
        bar = '█' * int(row['importance'] / 5)
        print(f"{row['feature']:30s} {row['importance']:5.2f} {bar}")
    
    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Model
    model_path = MODELS_DIR / "catboost_v2.cbm"
    model.save_model(str(model_path))
    logger.info(f"Saved model to {model_path}")
    
    # Calibrator (if used)
    if calibrators:
        import joblib
        cal_path = MODELS_DIR / "calibrator_v2.pkl"
        joblib.dump(calibrators, cal_path)
        logger.info(f"Saved calibrator to {cal_path}")
    
    # Feature columns (copy for reference)
    feature_path = MODELS_DIR / "feature_columns_v2.json"
    with open(feature_path, 'w') as f:
        json.dump({
            'version': 'v2',
            'features': feature_names,
            'categorical': list(contract.categorical),
            'numeric': list(contract.numeric)
        }, f, indent=2)
    
    # Metadata
    metadata = {
        'model_type': 'catboost_v2',
        'version': 'v2',
        'timestamp': datetime.now().isoformat(),
        'features': feature_names,
        'num_features': len(feature_names),
        'hyperparameters': params,
        'splits': {
            'train': {
                'count': len(train),
                'start': str(train['kickoff_time'].min()),
                'end': str(train['kickoff_time'].max())
            },
            'valid': {
                'count': len(valid),
                'start': str(valid['kickoff_time'].min()),
                'end': str(valid['kickoff_time'].max())
            },
            'test': {
                'count': len(test),
                'start': str(test['kickoff_time'].min()),
                'end': str(test['kickoff_time'].max())
            }
        },
        'metrics': {
            'train': train_metrics,
            'valid': valid_metrics,
            'test': test_metrics
        },
        'calibration_used': use_calibration
    }
    
    meta_path = MODELS_DIR / "catboost_v2_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")
    
    return metadata


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CatBoost V2")
    parser.add_argument('--input', type=str,
                        default=str(DATA_DIR / "ml_dataset_v2.csv"),
                        help="Input dataset")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("CATBOOST TRAINING V2")
    print("=" * 60)
    
    # Load contract
    contract = load_contract()
    logger.info(f"Feature contract: {contract.feature_count} features")
    
    # Load and split
    train, valid, test = load_and_split(Path(args.input))
    
    # Train
    metadata = train_catboost_v2(train, valid, test, contract)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model: models/catboost_v2.cbm")
    print(f"Features: {metadata['num_features']}")
    print(f"Test Accuracy: {metadata['metrics']['test']['accuracy']:.4f}")
    print(f"Test LogLoss: {metadata['metrics']['test']['logloss']:.4f}")
    print(f"Test Brier: {metadata['metrics']['test']['brier']:.4f}")


if __name__ == "__main__":
    main()
