#!/usr/bin/env python3
"""
Probability Calibration (Task 6)

Fits isotonic calibration on validation predictions and saves calibrator.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
import json
import joblib
import logging

from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.snapshot_config import FEATURE_ORDER, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.models.snapshot_calibrator import MultiClassCalibrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
DATASET_PATH = DATA_DIR / "snapshot_dataset.csv"
MODEL_PATH = MODELS_DIR / "catboost_snapshot.cbm"
CALIBRATOR_PATH = MODELS_DIR / "calibrator.pkl"


def load_data_and_model():
    """Load dataset and trained model."""
    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    
    # Load model
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    
    return df, model


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Extract feature matrix and labels."""
    X = df[FEATURE_ORDER].copy()
    y = df['label'].values
    
    # Fill NaN in numeric columns
    for col in NUMERIC_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna(0.0)
    
    # Fill NaN in categorical columns
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna('Unknown').astype(str)
    
    return X, y


def compute_reliability_table(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute reliability table for all classes."""
    class_names = ['Home', 'Draw', 'Away']
    all_rows = []
    
    for c, class_name in enumerate(class_names):
        probs = y_proba[:, c]
        actual = (y_true == c).astype(float)
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            low = bin_edges[i]
            high = bin_edges[i + 1]
            mask = (probs >= low) & (probs < high)
            
            if mask.sum() > 0:
                mean_pred = probs[mask].mean()
                mean_actual = actual[mask].mean()
                count = mask.sum()
                gap = abs(mean_pred - mean_actual)
            else:
                mean_pred = (low + high) / 2
                mean_actual = np.nan
                count = 0
                gap = np.nan
            
            all_rows.append({
                'class': class_name,
                'bin': f"{low:.1f}-{high:.1f}",
                'predicted': mean_pred,
                'actual': mean_actual,
                'count': count,
                'gap': gap,
            })
    
    return pd.DataFrame(all_rows)


def main():
    """Main calibration pipeline."""
    print("\n" + "=" * 60)
    print("PROBABILITY CALIBRATION")
    print("=" * 60)
    
    # Load data
    df, model = load_data_and_model()
    logger.info(f"Loaded {len(df)} samples and model")
    
    # Split for calibration (use validation set)
    train_end = "2023-06-30"
    val_end = "2024-01-31"
    
    val_df = df[(df['kickoff_time'] >= train_end) & (df['kickoff_time'] < val_end)]
    test_df = df[df['kickoff_time'] >= val_end]
    
    logger.info(f"Validation: {len(val_df)} samples")
    logger.info(f"Test: {len(test_df)} samples")
    
    # Prepare features
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)
    
    # Get raw predictions
    y_val_proba = model.predict_proba(X_val)
    y_test_proba_raw = model.predict_proba(X_test)
    
    # Fit calibrator
    print("\n🔧 Fitting calibrator on validation set...")
    calibrator = MultiClassCalibrator()
    calibrator.fit(y_val, y_val_proba)
    
    # Apply calibration to test set
    y_test_proba_cal = calibrator.predict_proba(y_test_proba_raw)
    
    # Compute metrics before/after
    from sklearn.metrics import log_loss
    
    ll_raw = log_loss(y_test, y_test_proba_raw, labels=[0, 1, 2])
    ll_cal = log_loss(y_test, y_test_proba_cal, labels=[0, 1, 2])
    
    print(f"\n📊 Test Set Results:")
    print(f"  LogLoss (raw):        {ll_raw:.4f}")
    print(f"  LogLoss (calibrated): {ll_cal:.4f}")
    print(f"  Improvement:          {(ll_raw - ll_cal):.4f} ({(ll_raw - ll_cal) / ll_raw * 100:.2f}%)")
    
    # Reliability tables
    print("\n📈 RELIABILITY TABLE (Calibrated)")
    reliability = compute_reliability_table(y_test, y_test_proba_cal)
    
    for class_name in ['Home', 'Draw', 'Away']:
        class_table = reliability[reliability['class'] == class_name]
        print(f"\n{class_name}:")
        print(class_table[['bin', 'predicted', 'actual', 'count', 'gap']].to_string(index=False))
    
    # Compute ECE (Expected Calibration Error)
    total_samples = reliability['count'].sum()
    ece = (reliability['count'] * reliability['gap'].fillna(0)).sum() / total_samples
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    
    # Save calibrator
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, CALIBRATOR_PATH)
    logger.info(f"Saved calibrator to {CALIBRATOR_PATH}")
    
    # Save reliability table
    reliability_path = MODELS_DIR / "reliability_table.csv"
    reliability.to_csv(reliability_path, index=False)
    logger.info(f"Saved reliability table to {reliability_path}")
    
    print("\n✅ CALIBRATION COMPLETE")


if __name__ == "__main__":
    main()
