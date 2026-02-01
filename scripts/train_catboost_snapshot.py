#!/usr/bin/env python3
"""
CatBoost Snapshot Training (Task 5)

Trains CatBoost model on snapshot dataset with walk-forward validation.
Outputs model, feature columns, and detailed metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import logging
import joblib

from catboost import CatBoostClassifier, Pool

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.snapshot_config import FEATURE_ORDER, CATEGORICAL_FEATURES, NUMERIC_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
DATASET_PATH = DATA_DIR / "snapshot_dataset.csv"


def load_dataset() -> pd.DataFrame:
    """Load snapshot dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Run: python scripts/build_snapshot_dataset.py"
        )
    
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Loaded {len(df)} samples")
    return df


def time_split(
    df: pd.DataFrame,
    train_end: str = "2023-06-30",
    val_end: str = "2024-01-31",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward time split.
    
    Default split:
    - Train: everything before 2023-07-01
    - Valid: 2023-07-01 to 2024-01-31
    - Test: 2024-02-01 onwards
    """
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    
    train = df[df['kickoff_time'] < train_end].copy()
    val = df[(df['kickoff_time'] >= train_end) & (df['kickoff_time'] < val_end)].copy()
    test = df[df['kickoff_time'] >= val_end].copy()
    
    logger.info(f"Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    logger.info(f"  Train: {train['kickoff_time'].min()} to {train['kickoff_time'].max()}")
    logger.info(f"  Val:   {val['kickoff_time'].min()} to {val['kickoff_time'].max()}")
    logger.info(f"  Test:  {test['kickoff_time'].min()} to {test['kickoff_time'].max()}")
    
    return train, val, test


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


def train_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Dict = None,
) -> CatBoostClassifier:
    """Train CatBoost model."""
    
    # Default params
    default_params = {
        'loss_function': 'MultiClass',
        'iterations': 1000,
        'depth': 6,
        'learning_rate': 0.03,
        'l2_leaf_reg': 5,
        'early_stopping_rounds': 50,
        'random_seed': 42,
        'verbose': 100,
    }
    
    if params:
        default_params.update(params)
    
    # Get categorical feature indices
    cat_indices = [FEATURE_ORDER.index(f) for f in CATEGORICAL_FEATURES if f in FEATURE_ORDER]
    
    # Create pools
    train_pool = Pool(
        X_train, 
        label=y_train,
        cat_features=cat_indices,
        feature_names=list(X_train.columns)
    )
    val_pool = Pool(
        X_val,
        label=y_val,
        cat_features=cat_indices,
        feature_names=list(X_val.columns)
    )
    
    # Train
    model = CatBoostClassifier(**default_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    
    return model


def evaluate_model(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    y: np.ndarray,
    split_name: str = "Test"
) -> Dict[str, Any]:
    """Evaluate model and return metrics."""
    from sklearn.metrics import accuracy_score, log_loss
    
    # Predictions
    y_proba = model.predict_proba(X)
    y_pred = model.predict(X).flatten().astype(int)
    
    # Metrics
    acc = accuracy_score(y, y_pred)
    logloss = log_loss(y, y_proba, labels=[0, 1, 2])
    
    # Brier score (multi-class)
    y_onehot = np.zeros((len(y), 3))
    for i, label in enumerate(y):
        if 0 <= label <= 2:
            y_onehot[i, label] = 1
    brier = np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1))
    
    # Per-class Brier
    brier_per_class = []
    for c in range(3):
        mask = y == c
        if mask.sum() > 0:
            brier_c = np.mean((y_proba[mask, c] - 1) ** 2)
        else:
            brier_c = 0.0
        brier_per_class.append(brier_c)
    
    metrics = {
        'accuracy': acc,
        'log_loss': logloss,
        'brier': brier,
        'brier_home': brier_per_class[0],
        'brier_draw': brier_per_class[1],
        'brier_away': brier_per_class[2],
        'n_samples': len(y),
    }
    
    logger.info(f"\n{split_name} Metrics:")
    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info(f"  LogLoss:  {logloss:.4f}")
    logger.info(f"  Brier:    {brier:.4f}")
    
    return metrics


def compute_calibration_table(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    class_idx: int = 0,
    class_name: str = "Home"
) -> pd.DataFrame:
    """Compute calibration table for a specific class."""
    probs = y_proba[:, class_idx]
    actual = (y_true == class_idx).astype(float)
    
    # Bin edges
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    rows = []
    for i in range(n_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]
        mask = (probs >= low) & (probs < high)
        
        if mask.sum() > 0:
            mean_pred = probs[mask].mean()
            mean_actual = actual[mask].mean()
            count = mask.sum()
        else:
            mean_pred = (low + high) / 2
            mean_actual = np.nan
            count = 0
        
        rows.append({
            'bin': f"{low:.1f}-{high:.1f}",
            'mean_pred': mean_pred,
            'mean_actual': mean_actual,
            'count': count,
            'class': class_name,
        })
    
    return pd.DataFrame(rows)


def save_model(
    model: CatBoostClassifier,
    train_metrics: Dict,
    val_metrics: Dict,
    test_metrics: Dict,
    calibration_tables: Dict,
):
    """Save model and artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = MODELS_DIR / "catboost_snapshot.cbm"
    model.save_model(str(model_path))
    logger.info(f"Saved model to {model_path}")
    
    # Save feature columns
    columns_path = MODELS_DIR / "feature_columns.json"
    with open(columns_path, 'w') as f:
        json.dump({
            "feature_columns": FEATURE_ORDER,
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "version": "1.0.0",
            "created_at": timestamp,
        }, f, indent=2)
    
    # Save metrics
    metadata = {
        "model_type": "catboost_snapshot",
        "version": "1.0.0",
        "created_at": timestamp,
        "feature_count": len(FEATURE_ORDER),
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
        },
        "feature_importance": dict(zip(
            model.feature_names_,
            model.feature_importances_.tolist()
        )),
    }
    
    metadata_path = MODELS_DIR / "catboost_snapshot_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save calibration tables
    for class_name, table in calibration_tables.items():
        table_path = MODELS_DIR / f"calibration_{class_name.lower()}.csv"
        table.to_csv(table_path, index=False)
    
    logger.info(f"Saved all artifacts to {MODELS_DIR}")


def print_feature_importance(model: CatBoostClassifier, top_n: int = 15):
    """Print top feature importances."""
    importance = dict(zip(model.feature_names_, model.feature_importances_))
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
    
    print("\n" + "=" * 50)
    print(f"TOP {top_n} FEATURE IMPORTANCES")
    print("=" * 50)
    for i, (feat, imp) in enumerate(sorted_imp[:top_n], 1):
        bar = "█" * int(imp / 2)
        print(f"{i:2}. {feat:<25} {imp:6.2f} {bar}")


def main():
    """Main training pipeline."""
    print("\n" + "=" * 60)
    print("CATBOOST SNAPSHOT TRAINING")
    print("=" * 60)
    
    # Load data
    df = load_dataset()
    
    # Split
    train_df, val_df, test_df = time_split(df)
    
    # Prepare features
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)
    
    logger.info(f"Feature matrix shape: {X_train.shape}")
    
    # Train
    print("\n🚀 Training CatBoost...")
    model = train_catboost(X_train, y_train, X_val, y_val)
    
    # Evaluate
    print("\n📊 Evaluating...")
    train_metrics = evaluate_model(model, X_train, y_train, "Train")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")
    
    # Calibration tables
    print("\n📈 Computing calibration tables...")
    y_test_proba = model.predict_proba(X_test)
    
    calibration_tables = {
        'Home': compute_calibration_table(y_test, y_test_proba, class_idx=0, class_name="Home"),
        'Draw': compute_calibration_table(y_test, y_test_proba, class_idx=1, class_name="Draw"),
        'Away': compute_calibration_table(y_test, y_test_proba, class_idx=2, class_name="Away"),
    }
    
    # Print calibration
    for class_name, table in calibration_tables.items():
        print(f"\nCalibration Table ({class_name}):")
        print(table[['bin', 'mean_pred', 'mean_actual', 'count']].to_string(index=False))
    
    # Feature importance
    print_feature_importance(model)
    
    # Save
    save_model(model, train_metrics, val_metrics, test_metrics, calibration_tables)
    
    print("\n✅ TRAINING COMPLETE")
    print(f"Model saved to: {MODELS_DIR / 'catboost_snapshot.cbm'}")


if __name__ == "__main__":
    main()
