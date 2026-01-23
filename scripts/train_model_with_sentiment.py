"""
Train CatBoost model with sentiment features (28 total features).

Enhanced version of train_model.py specifically for sentiment-augmented dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import logging
from datetime import datetime

from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    confusion_matrix
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def time_based_split(df, train_frac=0.70, val_frac=0.15):
    """Split data by time."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    logger.info(f"Time-based split:")
    logger.info(f"  Train: {len(train)} matches")
    logger.info(f"  Val:   {len(val)} matches")
    logger.info(f"  Test:  {len(test)} matches")
    
    return train, val, test


def prepare_data(df, feature_cols):
    """Prepare X and y."""
    X = df[feature_cols].values
    
    # Encode target: H=0, D=1, A=2
    result_map = {'H': 0, 'D': 1, 'A': 2}
    y = df['FTR'].map(result_map).values
    
    return X, y


def train_catboost(X_train, y_train, X_val, y_val):
    """Train CatBoost."""
    logger.info("Training CatBoost with sentiment features...")
    
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        early_stopping_rounds=50,
        random_seed=42,
        verbose=False,  # Less verbose for cleaner output
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    
    logger.info(f"✓ Training complete (iterations: {model.best_iteration_})")
    return model


def calibrate_model(model, X_val, y_val):
    """Calibrate probabilities."""
    logger.info("Calibrating probabilities...")
    
    calibrator = CalibratedClassifierCV(
        model,
        method='isotonic',
        cv='prefit'
    )
    calibrator.fit(X_val, y_val)
    
    logger.info("✓ Calibration complete")
    return calibrator


def evaluate_model(model, X, y, name="Test"):
    """Evaluate model."""
    logger.info(f"\n{name} Set:")
    
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    acc = accuracy_score(y, y_pred)
    logloss = log_loss(y, y_proba)
    
    # Brier score
    brier_scores = []
    for i in range(3):
        y_binary = (y == i).astype(int)
        brier = brier_score_loss(y_binary, y_proba[:, i])
        brier_scores.append(brier)
    brier_avg = np.mean(brier_scores)
    
    logger.info(f"  Accuracy:    {acc:.4f} ({acc:.2%})")
    logger.info(f"  Log Loss:    {logloss:.4f}")
    logger.info(f"  Brier Score: {brier_avg:.4f}")
    
    return {
        'accuracy': float(acc),
        'log_loss': float(logloss),
        'brier_score': float(brier_avg),
    }


def get_feature_importance(model, feature_names, top_k=10):
    """Get feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    logger.info(f"\nTop {top_k} Most Important Features:")
    for i, idx in enumerate(indices[:top_k], 1):
        logger.info(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Check sentiment features
    sentiment_features = [
        'home_sentiment_score', 'away_sentiment_score',
        'home_injury_flag', 'away_injury_flag',
        'home_news_volume', 'away_news_volume'
    ]
    
    logger.info(f"\nSentiment Feature Importances:")
    for feat in sentiment_features:
        if feat in feature_names:
            idx = feature_names.index(feat)
            importance = importances[idx]
            rank = list(indices).index(idx) + 1
            logger.info(f"  {feat}: {importance:.4f} (rank {rank}/{len(feature_names)})")


def save_model_sentiment(model, calibrator, scaler, feature_names, metrics):
    """Save sentiment-enhanced model."""
    output_dir = Path('models')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save with _sentiment suffix
    model_file = output_dir / f'catboost_sentiment_v1_{timestamp}.pkl'
    calib_file = output_dir / f'calibrator_sentiment_v1_{timestamp}.pkl'
    scaler_file = output_dir / f'scaler_sentiment_v1_{timestamp}.pkl'
    
    joblib.dump(model, model_file)
    joblib.dump(calibrator, calib_file)
    joblib.dump(scaler, scaler_file)
    
    logger.info(f"\n✓ Saved model: {model_file.name}")
    logger.info(f"✓ Saved calibrator: {calib_file.name}")
    logger.info(f"✓ Saved scaler: {scaler_file.name}")
    
    # Metadata
    metadata = {
        'model_type': 'catboost_with_sentiment',
        'version': 'v1',
        'timestamp': timestamp,
        'train_date': datetime.now().isoformat(),
        'features': feature_names,
        'num_features': len(feature_names),
        'sentiment_features': 6,
        'base_features': len(feature_names) - 6,
        'metrics': metrics,
        'training_data': {
            'source': 'Football-Data.co.uk + Mock Sentiment',
            'league': 'EPL',
            'seasons': '2021-22, 2022-23, 2023-24',
        }
    }
    
    metadata_file = output_dir / f'metadata_sentiment_v1_{timestamp}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Saved metadata: {metadata_file.name}")
    
    # Create symlinks
    for old_file, new_name in [
        (model_file, 'catboost_sentiment_v1_latest.pkl'),
        (calib_file, 'calibrator_sentiment_v1_latest.pkl'),
        (scaler_file, 'scaler_sentiment_v1_latest.pkl'),
        (metadata_file, 'metadata_sentiment_v1_latest.json'),
    ]:
        symlink = output_dir / new_name
        if symlink.exists():
            symlink.unlink()
        symlink.symlink_to(old_file.name)
    
    return model_file


def main():
    """Main training pipeline."""
    logger.info("=" * 70)
    logger.info("CATBOOST WITH SENTIMENT FEATURES")
    logger.info("=" * 70)
    
    # Load sentiment-enhanced dataset
    data_file = Path('data/processed/epl_features_with_sentiment_2021_2024.csv')
    
    logger.info(f"\nLoading data: {data_file}")
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Define features (all except metadata columns)
    feature_cols = [col for col in df.columns 
                   if col not in ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'FTR']]
    
    logger.info(f"Total features: {len(feature_cols)}")
    logger.info(f"  Base features: 22")
    logger.info(f"  Sentiment features: 6")
    
    # Split
    train_df, val_df, test_df = time_based_split(df)
    
    X_train, y_train = prepare_data(train_df, feature_cols)
    X_val, y_val = prepare_data(val_df, feature_cols)
    X_test, y_test = prepare_data(test_df, feature_cols)
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Train
    model = train_catboost(X_train, y_train, X_val, y_val)
    
    # Feature importance
    get_feature_importance(model, feature_cols, top_k=15)
    
    # Calibrate
    calibrator = calibrate_model(model, X_val, y_val)
    
    # Evaluate
    metrics_val = evaluate_model(model, X_val, y_val, "Validation (Uncalibrated)")
    metrics_test_uncalib = evaluate_model(model, X_test, y_test, "Test (Uncalibrated)")
    metrics_test = evaluate_model(calibrator, X_test, y_test, "Test (Calibrated)")
    
    # Save
    save_model_sentiment(
        model, calibrator, scaler, 
        feature_cols,
        {'val': metrics_val, 'test_uncalibrated': metrics_test_uncalib, 'test': metrics_test}
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Test Accuracy:    {metrics_test['accuracy']:.2%}")
    logger.info(f"Test Brier Score: {metrics_test['brier_score']:.4f}")
    logger.info(f"Test Log Loss:    {metrics_test['log_loss']:.4f}")
    
    # Compare to baseline (if available)
    logger.info("\nComparison to baseline (22 features):")
    logger.info("  Baseline accuracy: ~60.23%")
    logger.info("  Baseline Brier: ~0.1825")
    diff = metrics_test['accuracy'] - 0.6023
    logger.info(f"  Difference: {diff:+.2%}")


if __name__ == '__main__':
    main()
