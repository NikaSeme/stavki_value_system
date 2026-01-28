#!/usr/bin/env python3
"""
Train simple CatBoost model on multi-league data using only team and league identities.

This is a baseline model that demonstrates multi-league training without requiring
complex feature engineering. It only uses:
- HomeTeam (categorical)
- AwayTeam (categorical)  
- League (categorical)

This tests whether CatBoost can learn league-specific patterns automatically.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import logging
from datetime import datetime

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.calibration import get_best_calibrator

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


def prepare_simple_features(df):
    """
    Prepare features = [HomeTeam, AwayTeam, League]
    Target = FTR mapped to {H:0, D:1, A:2}
    """
    features = ['HomeTeam', 'AwayTeam', 'League']
    X = df[features]
    
    result_map = {'H': 0, 'D': 1, 'A': 2}
    y = df['FTR'].map(result_map).values
    
    return X, y


def main():
    logger.info("=" * 70)
    logger.info("SIMPLE MULTI-LEAGUE CATBOOST (Team + League Only)")
    logger.info("=" * 70)
    
    # Load multi-league data
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / 'data' / 'processed' / 'multi_league_clean_2021_2024.csv'
    
    if not data_file.exists():
        logger.error(f"Data missing: {data_file}")
        return
        
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    logger.info(f"Loaded {len(df)} matches")
    logger.info(f"Leagues: {df['League'].value_counts().to_dict()}")
    
    # Split
    train_df, val_df, test_df = time_based_split(df)
    
    X_train, y_train = prepare_simple_features(train_df)
    X_val, y_val = prepare_simple_features(val_df)
    X_test, y_test = prepare_simple_features(test_df)
    
    # Train CatBoost with categorical features
    logger.info("Training CatBoost...")
    
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='MultiClass',
        cat_features=[0, 1, 2],  # All 3 features are categorical
        random_seed=42,
        verbose=False
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)
    
    # Evaluate
    logger.info("\\nEvaluating...")
    y_pred_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)
    
    # Brier score (one-vs-rest avg)
    brier = np.mean([
        brier_score_loss((y_test == i).astype(int), y_pred_proba[:, i])
        for i in range(3)
    ])
    
    logger.info(f"Test Accuracy: {acc:.2%}")
    logger.info(f"Test Log Loss: {logloss:.4f}")
    logger.info(f"Test Brier Score: {brier:.4f}")
    
    # Calibrate
    logger.info("\\nCalibrating...")
    calibrator = get_best_calibrator(model)
    calibrator.fit(X_val, y_val)
    
    # Evaluate calibrated
    y_cal_proba = calibrator.predict_proba(X_test)
    y_cal_pred = np.argmax(y_cal_proba, axis=1)
    
    cal_acc = accuracy_score(y_test, y_cal_pred)
    cal_logloss = log_loss(y_test, y_cal_proba)
    cal_brier = np.mean([
        brier_score_loss((y_test == i).astype(int), y_cal_proba[:, i])
        for i in range(3)
    ])
    
    logger.info(f"Calibrated Test Accuracy: {cal_acc:.2%}")
    logger.info(f"Calibrated Test Log Loss: {cal_logloss:.4f}")
    logger.info(f"Calibrated Test Brier Score: {cal_brier:.4f}")
    
    # Save
    output_dir = base_dir / 'models'
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_file = output_dir / f'catboost_multi_league_{timestamp}.pkl'
    calib_file = output_dir / f'calibrator_multi_league_{timestamp}.pkl'
    
    joblib.dump(model, model_file)
    joblib.dump(calibrator, calib_file)
    
    metadata = {
        'model_type': 'catboost_simple_multileague',
        'version': 'v6.5_ml',
        'timestamp': timestamp,
        'train_date': datetime.now().isoformat(),
        'features': ['HomeTeam', 'AwayTeam', 'League'],
        'leagues': df['League'].unique().tolist(),
        'num_matches': len(df),
        'metrics': {
            'test_accuracy': cal_acc,
            'test_logloss': cal_logloss,
            'test_brier': cal_brier
        }
    }
    
    metadata_file = output_dir / f'metadata_multi_league_{timestamp}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create symlinks
    for old_file, new_name in [
        (model_file, 'catboost_multi_league_latest.pkl'),
        (calib_file, 'calibrator_multi_league_latest.pkl'),
        (metadata_file, 'metadata_multi_league_latest.json'),
    ]:
        symlink = output_dir / new_name
        if symlink.is_symlink() or symlink.exists():
            symlink.unlink()
        symlink.symlink_to(old_file.name)
    
    logger.info(f"\\n✅ Model saved to {model_file}")
    logger.info("✅ TRAINING COMPLETE")


if __name__ == '__main__':
    main()
