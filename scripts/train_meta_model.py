"""
Train meta-model for optimal ensemble weighting.

Uses Logistic Regression to learn how to best combine
Poisson, CatBoost, and Neural model predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
import logging
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.utils import FrozenEstimator



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_meta_data():
    """Load compiled meta-datasets."""
    data_dir = Path('data/meta')
    
    X_train = np.load(data_dir / 'X_meta_train.npy')
    y_train = np.load(data_dir / 'y_meta_train.npy')
    
    X_val = np.load(data_dir / 'X_meta_val.npy')
    y_val = np.load(data_dir / 'y_meta_val.npy')
    
    X_test = np.load(data_dir / 'X_meta_test.npy')
    y_test = np.load(data_dir / 'y_meta_test.npy')
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_meta_model(X_train, y_train, X_val, y_val):
    """Train Logistic Regression meta-model."""
    logger.info("Training meta-model (Logistic Regression)...")
    
    meta_model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    
    meta_model.fit(X_train, y_train)
    
    logger.info("✓ Meta-model trained")
    
    # Analyze learned weights
    logger.info("\nLearned model weights:")
    logger.info(f"  Coefficients shape: {meta_model.coef_.shape}")  # (3 classes, 9 features)
    
    feature_names = [
        'poisson_H', 'poisson_D', 'poisson_A',
        'catboost_H', 'catboost_D', 'catboost_A',
        'neural_H', 'neural_D', 'neural_A'
    ]
    
    # Average absolute coefficients per model
    poisson_weight = np.mean(np.abs(meta_model.coef_[:, 0:3]))
    catboost_weight = np.mean(np.abs(meta_model.coef_[:, 3:6]))
    neural_weight = np.mean(np.abs(meta_model.coef_[:, 6:9]))
    
    total_weight = poisson_weight + catboost_weight + neural_weight
    
    logger.info(f"\n  Model importance (normalized):")
    logger.info(f"    Poisson:  {poisson_weight/total_weight:.1%}")
    logger.info(f"    CatBoost: {catboost_weight/total_weight:.1%}")
    logger.info(f"    Neural:   {neural_weight/total_weight:.1%}")
    
    return meta_model


def calibrate_meta_model(meta_model, X_val, y_val):
    """Apply isotonic calibration to meta-model."""
    logger.info("\nCalibrating meta-model...")
    
    # Using Scikit-Learn 1.6+ FrozenEstimator for the professional long-term solution
    calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(meta_model),
        method='isotonic',
        cv='prefit'
    )
    calibrator.fit(X_val, y_val)
    
    logger.info("✓ Calibration complete")
    return calibrator


def evaluate_model(model, X, y, name="Test"):
    """Evaluate model performance."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    acc = accuracy_score(y, y_pred)
    logloss = log_loss(y, y_proba)
    
    # Brier score (average across classes)
    brier_scores = []
    for i in range(3):
        y_binary = (y == i).astype(int)
        brier = brier_score_loss(y_binary, y_proba[:, i])
        brier_scores.append(brier)
    brier_avg = np.mean(brier_scores)
    
    logger.info(f"\n{name} Set:")
    logger.info(f"  Accuracy:    {acc:.4f} ({acc:.2%})")
    logger.info(f"  Log Loss:    {logloss:.4f}")
    logger.info(f"  Brier Score: {brier_avg:.4f}")
    
    return {
        'accuracy': float(acc),
        'log_loss': float(logloss),
        'brier_score': float(brier_avg)
    }


def compute_baseline(X, y):
    """Compute simple averaging baseline."""
    # Extract predictions from meta-features
    poisson_probs = X[:, 0:3]  # [:, 0] = away, [:, 1] = draw, [:, 2] = home
    catboost_probs = X[:, 3:6]
    neural_probs = X[:, 6:9]
    
    # Simple average
    avg_probs = (poisson_probs + catboost_probs + neural_probs) / 3.0
    
    # Predictions
    y_pred = np.argmax(avg_probs, axis=1)
    
    # Metrics
    acc = accuracy_score(y, y_pred)
    logloss = log_loss(y, avg_probs)
    
    brier_scores = []
    for i in range(3):
        y_binary = (y == i).astype(int)
        brier = brier_score_loss(y_binary, avg_probs[:, i])
        brier_scores.append(brier)
    brier_avg = np.mean(brier_scores)
    
    return {
        'accuracy': float(acc),
        'log_loss': float(logloss),
        'brier_score': float(brier_avg)
    }


def save_meta_model(meta_model, calibrator, metrics):
    """Save meta-model artifacts."""
    output_dir = Path('models')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save meta-model
    model_file = output_dir / f'meta_model_v1_{timestamp}.pkl'
    joblib.dump(meta_model, model_file)
    logger.info(f"\n✓ Saved meta-model: {model_file.name}")
    
    # Save calibrator
    calib_file = output_dir / f'meta_calibrator_v1_{timestamp}.pkl'
    joblib.dump(calibrator, calib_file)
    logger.info(f"✓ Saved calibrator: {calib_file.name}")
    
    # Save metadata
    metadata = {
        'model_type': 'logistic_regression_meta',
        'version': 'v1',
        'timestamp': timestamp,
        'train_date': datetime.now().isoformat(),
        'base_models': ['poisson', 'catboost', 'neural'],
        'num_meta_features': 9,
        'metrics': metrics,
        'calibration': 'isotonic'
    }
    
    metadata_file = output_dir / f'meta_metadata_v1_{timestamp}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Saved metadata: {metadata_file.name}")
    
    # Create symlinks
    for old_file, new_name in [
        (model_file, 'meta_model_v1_latest.pkl'),
        (calib_file, 'meta_calibrator_v1_latest.pkl'),
        (metadata_file, 'meta_metadata_v1_latest.json')
    ]:
        symlink = output_dir / new_name
        if symlink.exists():
            symlink.unlink()
        symlink.symlink_to(old_file.name)
    
    logger.info("✓ Created symlinks")


def main():
    """Main training pipeline."""
    logger.info("=" * 70)
    logger.info("META-MODEL TRAINING")
    logger.info("=" * 70)
    
    # Load data
    logger.info("\nLoading meta-datasets...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_meta_data()
    
    logger.info(f"  Train: {X_train.shape}")
    logger.info(f"  Val:   {X_val.shape}")
    logger.info(f"  Test:  {X_test.shape}")
    
    # Train meta-model
    meta_model = train_meta_model(X_train, y_train, X_val, y_val)
    
    # Calibrate
    calibrator = calibrate_meta_model(meta_model, X_val, y_val)
    
    # Evaluate
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE (Simple Averaging)")
    logger.info("=" * 70)
    
    baseline_test = compute_baseline(X_test, y_test)
    logger.info(f"\nTest Set (Baseline):")
    logger.info(f"  Accuracy:    {baseline_test['accuracy']:.4f}")
    logger.info(f"  Brier Score: {baseline_test['brier_score']:.4f}")
    logger.info(f"  Log Loss:    {baseline_test['log_loss']:.4f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("META-MODEL (Learned Weighting)")
    logger.info("=" * 70)
    
    meta_val = evaluate_model(meta_model, X_val, y_val, "Validation (Uncalibrated)")
    meta_test_uncalib = evaluate_model(meta_model, X_test, y_test, "Test (Uncalibrated)")
    meta_test = evaluate_model(calibrator, X_test, y_test, "Test (Calibrated)")
    
    # Comparison
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON")
    logger.info("=" * 70)
    
    logger.info(f"\n{'Metric':<20} {'Baseline':<12} {'Meta-Model':<12} {'Improvement'}")
    logger.info("-" * 60)
    
    acc_diff = (meta_test['accuracy'] - baseline_test['accuracy']) * 100
    brier_diff = (baseline_test['brier_score'] - meta_test['brier_score']) * 100
    logloss_diff = (baseline_test['log_loss'] - meta_test['log_loss']) * 100
    
    logger.info(f"{'Accuracy':<20} {baseline_test['accuracy']:.4f}      {meta_test['accuracy']:.4f}      {acc_diff:+.2f} pp")
    logger.info(f"{'Brier Score':<20} {baseline_test['brier_score']:.4f}      {meta_test['brier_score']:.4f}      {brier_diff:+.4f}")
    logger.info(f"{'Log Loss':<20} {baseline_test['log_loss']:.4f}      {meta_test['log_loss']:.4f}      {logloss_diff:+.4f}")
    
    # Save
    metrics = {
        'baseline': baseline_test,
        'meta_val': meta_val,
        'meta_test_uncalibrated': meta_test_uncalib,
        'meta_test': meta_test
    }
    
    save_meta_model(meta_model, calibrator, metrics)
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ META-MODEL TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Test Accuracy: {meta_test['accuracy']:.2%}")
    logger.info(f"Test Brier:    {meta_test['brier_score']:.4f}")
    logger.info(f"Improvement:   {acc_diff:+.2f}pp accuracy, {brier_diff:+.4f} Brier")


if __name__ == '__main__':
    main()
