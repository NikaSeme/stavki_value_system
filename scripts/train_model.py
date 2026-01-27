
"""
Train Optimized CatBoost model with Categorical Features and Hyperparameter Tuning.

Features:
- Includes HomeTeam/AwayTeam as categorical features (drastic improvement)
- Uses ColumnTransformer to handle mixed types (numeric vs string)
- Implements Randomized Search for hyperparameter optimization
- Saves best model found
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import logging
import sys
import sklearn
from datetime import datetime
import random
import itertools

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.calibration import get_best_calibrator

from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    confusion_matrix
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def time_based_split(df, train_frac=0.70, val_frac=0.15):
    """Split data by time to avoid leakage."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    logger.info(f"Time-based split:")
    logger.info(f"  Train: {len(train)} matches ({train['Date'].min()} to {train['Date'].max()})")
    logger.info(f"  Val:   {len(val)} matches ({val['Date'].min()} to {val['Date'].max()})")
    logger.info(f"  Test:  {len(test)} matches ({test['Date'].min()} to {test['Date'].max()})")
    
    return train, val, test


def prepare_data(df, feature_cols):
    """
    Prepare X and y from df.
    Returns DataFrame for X to preserve column names/types for ColumnTransformer.
    """
    X = df[feature_cols].copy()
    
    # Encode target: H=0, D=1, A=2
    result_map = {'H': 0, 'D': 1, 'A': 2}
    y = df['FTR'].map(result_map).values
    
    return X, y


def evaluate_model(model, X, y, name="Test"):
    """Evaluate model performance."""
    # Predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Metrics
    acc = accuracy_score(y, y_pred)
    logloss = log_loss(y, y_proba)
    
    # Brier score (per class)
    brier_scores = []
    for i in range(3):
        y_binary = (y == i).astype(int)
        brier = brier_score_loss(y_binary, y_proba[:, i])
        brier_scores.append(brier)
    brier_avg = np.mean(brier_scores)
    
    return {
        'accuracy': float(acc),
        'log_loss': float(logloss),
        'brier_score': float(brier_avg),
        'brier_per_class': [float(b) for b in brier_scores],
    }


def hyperparameter_search(X_train, y_train, X_val, y_val, cat_features_indices, n_trials=10):
    """Run randomized search for best hyperparameters."""
    logger.info(f"Starting Hyperparameter Search ({n_trials} trials)...")
    
    param_grid = {
        'depth': [4, 5, 6, 7],
        'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'random_strength': [0.1, 1, 2],
        'bagging_temperature': [0, 1]
    }
    
    best_loss = float('inf')
    best_model = None
    best_params = {}
    
    # Generate random combinations
    keys = list(param_grid.keys())
    # Create extensive list of combinations and sample n_trials
    all_combinations = list(itertools.product(*[param_grid[k] for k in keys]))
    random.shuffle(all_combinations)
    trials = all_combinations[:n_trials]
    
    for i, values in enumerate(trials):
        params = dict(zip(keys, values))
        
        logger.info(f"Trial {i+1}/{n_trials}: {params}")
        
        model = CatBoostClassifier(
            iterations=800, # slightly lower for search speed
            loss_function='MultiClass',
            eval_metric='MultiClass',
            early_stopping_rounds=30,
            verbose=False,
            cat_features=cat_features_indices,
            **params
        )
        
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )
        
        # Validation score
        val_loss = model.get_best_score()['validation']['MultiClass']
        logger.info(f"  -> Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            best_params = params
            logger.info("  ⭐️ New Best!")
            
    logger.info(f"\nSearch Complete. Best Loss: {best_loss:.4f}")
    logger.info(f"Best Params: {best_params}")
    
    return best_model, best_params


def save_model_artifacts(model, calibrator, scaler, feature_names, metrics, output_dir, best_params):
    """Save all model artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_file = output_dir / f'catboost_v1_{timestamp}.pkl'
    joblib.dump(model, model_file)
    
    # Save calibrator
    calib_file = output_dir / f'calibrator_v1_{timestamp}.pkl'
    joblib.dump(calibrator, calib_file)
    
    # Save scaler (ColumnTransformer)
    scaler_file = output_dir / f'scaler_v1_{timestamp}.pkl'
    joblib.dump(scaler, scaler_file)
    
    # Save metadata
    metadata = {
        'model_type': 'catboost_optimized',
        'version': 'v1',
        'timestamp': timestamp,
        'train_date': datetime.now().isoformat(),
        'features': feature_names,
        'num_features': len(feature_names),
        'metrics': metrics,
        'hyperparameters': best_params,
        'pipeline_info': 'Includes Categorical Features + Opt Loop'
    }
    
    metadata_file = output_dir / f'metadata_v1_{timestamp}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    # Create symlinks to latest
    for old_file, new_name in [
        (model_file, 'catboost_v1_latest.pkl'),
        (calib_file, 'calibrator_v1_latest.pkl'),
        (scaler_file, 'scaler_v1_latest.pkl'),
        (metadata_file, 'metadata_v1_latest.json'),
    ]:
        symlink = output_dir / new_name
        if symlink.is_symlink() or symlink.exists():
            symlink.unlink()
        symlink.symlink_to(old_file.name)
    
    logger.info("✓ Artifacts saved and linked to '_latest'")
    
    return model_file


def main():
    logger.info("=" * 70)
    logger.info("CATBOOST OPTIMIZED TRAINING (Hyperparams + Categoricals)")
    logger.info("=" * 70)
    
    # Load data
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / 'data' / 'processed' / 'epl_features_2021_2024.csv'
    
    if not data_file.exists():
        logger.error(f"Data missing: {data_file}")
        return
        
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # --- FEATURE ENGINEERING ---
    # Define Numerical Features
    exclude_cols = ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'FTR']
    num_features = [col for col in df.columns if col not in exclude_cols]
    
    # Define Categorical Features (The Upgrade!)
    cat_features = ['HomeTeam', 'AwayTeam']
    
    # Final feature list (ORDER MATTERS)
    feature_cols = num_features + cat_features
    
    logger.info(f"Numerical Features: {len(num_features)}")
    logger.info(f"Categorical Features: {cat_features}")
    
    # Split
    train_df, val_df, test_df = time_based_split(df)
    
    X_train_raw, y_train = prepare_data(train_df, feature_cols)
    X_val_raw, y_val = prepare_data(val_df, feature_cols)
    X_test_raw, y_test = prepare_data(test_df, feature_cols)
    
    # --- PREPROCESSING ---
    # We use ColumnTransformer to scale numerics but pass strings through
    # This scaler object effectively replaces the old StandardScaler
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', 'passthrough', cat_features)
        ]
    )
    
    logger.info("Preprocessing features...")
    # fit on train, transform all
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # CatBoost receives numpy array. 
    # Categorical columns are at the END (because feature_cols = num + cat)
    # So indices are [len(num), len(num)+1]
    cat_indices = [len(num_features), len(num_features) + 1]
    
    # --- HYPERPARAMETER TUNING ---
    best_model, best_params = hyperparameter_search(
        X_train, y_train, X_val, y_val, 
        cat_features_indices=cat_indices,
        n_trials=12  # Run 12 random trials
    )
    
    # --- CALIBRATION ---
    logger.info("Calibrating best model...")
    calibrator = get_best_calibrator(best_model)
    calibrator.fit(X_val, y_val)
    
    # --- EVALUATION ---
    metrics_test = evaluate_model(calibrator, X_test, y_test, "Test Set")
    logger.info(f"\nFinal Test Brier Score: {metrics_test['brier_score']:.4f}")
    logger.info(f"Final Test Accuracy:    {metrics_test['accuracy']:.2%}")
    
    # --- SAVE ---
    output_dir = base_dir / 'models'
    save_model_artifacts(
        best_model, calibrator, preprocessor,
        feature_cols,
        {'test': metrics_test},
        output_dir,
        best_params
    )
    
    logger.info("\n✅ OPTIMIZED TRAINING COMPLETE")

if __name__ == '__main__':
    main()
