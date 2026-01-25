"""
Train ensemble meta-model (stacking).

Combines Poisson (Model A) and CatBoost (Model B) predictions
using logistic regression, then applies isotonic calibration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ModelLoader, EnsembleModel
from scripts.train_poisson_model import PoissonMatchPredictor
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_catboost_predictions(test_df):
    """Generate CatBoost predictions on test set."""
    logger.info("Loading CatBoost model...")
    
    loader = ModelLoader()
    loader.load_latest()
    
    # Load features from feature-engineered dataset
    feature_file = Path(__file__).parent.parent / 'data' / 'processed' / 'epl_features_2021_2024.csv'
    features_df = pd.read_csv(feature_file)
    features_df['Date'] = pd.to_datetime(features_df['Date'])
    
    # Match test set dates
    test_dates = set(test_df['Date'])
    test_features = features_df[features_df['Date'].isin(test_dates)].copy()
    
    # Get feature columns
    feature_cols = [col for col in test_features.columns 
                   if col not in ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'FTR']]
    
    X_test = test_features[feature_cols].values
    
    logger.info(f"Predicting with CatBoost on {len(X_test)} matches...")
    probs_catboost = loader.predict(X_test)
    
    # Create DataFrame
    catboost_df = pd.DataFrame({
        'Date': test_features['Date'].values,
        'prob_home_ml': probs_catboost[:, 0],
        'prob_draw_ml': probs_catboost[:, 1],
        'prob_away_ml': probs_catboost[:, 2],
    })
    
    return catboost_df


def main():
    """Train ensemble meta-model."""
    logger.info("=" * 70)
    logger.info("ENSEMBLE META-MODEL TRAINING (STACKING)")
    logger.info("=" * 70)
    
    base_dir = Path(__file__).parent.parent
    
    # Load historical data
    data_file = base_dir / 'data' / 'processed' / 'epl_historical_2021_2024.csv'
    logger.info(f"Loading data from {data_file}")
    
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Same split as base models
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"\nDataset split:")
    logger.info(f"  Train: {len(train_df)} (for base models)")
    logger.info(f"  Val:   {len(val_df)} (for meta-model training)")
    logger.info(f"  Test:  {len(test_df)} (for final evaluation)")
    
    # Load Poisson model
    logger.info("\nLoading Poisson model...")
    poisson_model = PoissonMatchPredictor.load(
        base_dir / 'models' / 'poisson_v1_latest.pkl'
    )
    
    # Generate predictions on validation set (for meta-model training)
    logger.info("\nGenerating Poisson predictions on validation set...")
    poisson_val = poisson_model.predict(val_df)
    poisson_val.columns = ['prob_home', 'prob_draw', 'prob_away']
    
    # Generate CatBoost predictions on validation set
    logger.info("Generating CatBoost predictions on validation set...")
    catboost_val = load_catboost_predictions(val_df)
    
    # Ensure both have same length (should be same val set)
    logger.info(f"Poisson predictions: {len(poisson_val)}")
    logger.info(f"CatBoost predictions: {len(catboost_val)}")
    
    # Use indices for alignment (both should match val_df)
    if len(poisson_val) != len(val_df) or len(catboost_val) != len(val_df):
        logger.error("Prediction length mismatch!")
        logger.error(f"Val set: {len(val_df)}, Poisson: {len(poisson_val)}, CatBoost: {len(catboost_val)}")
        raise ValueError("Prediction lengths don't match validation set")
    
    # Stack predictions side by side
    merged_val = pd.concat([
        poisson_val.reset_index(drop=True),
        catboost_val[['prob_home_ml', 'prob_draw_ml', 'prob_away_ml']].reset_index(drop=True)
    ], axis=1)
    
    logger.info(f"Merged validation predictions: {len(merged_val)} matches")
    
    # Get labels
    result_map = {'H': 2, 'D': 1, 'A': 0}  # Match sklearn convention
    y_val = val_df['FTR'].map(result_map).values
    
    # Train ensemble
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING META-MODEL")
    logger.info("=" * 70)
    
    ensemble = EnsembleModel(max_iter=1000, calibrate=True)
    
    stats = ensemble.train(
        poisson_probs=merged_val[['prob_home', 'prob_draw', 'prob_away']],
        ml_probs=merged_val[['prob_home_ml', 'prob_draw_ml', 'prob_away_ml']],
        y_true=y_val,
        calibration_split=0.3  # Use 30% of val set for calibration
    )
    
    logger.info("\nMeta-model training complete!")
    logger.info(f"  Training samples: {stats['train_samples']}")
    logger.info(f"  Training accuracy: {stats['train_accuracy']:.2%}")
    
    if 'calibration_samples' in stats:
        logger.info(f"  Calibration samples: {stats['calibration_samples']}")
        logger.info(f"  Calibration accuracy: {stats['calibration_accuracy']:.2%}")
    
    # Evaluate on test set
    logger.info("\n" + "=" * 70)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 70)
    
    # Generate predictions on test set
    logger.info("\nGenerating predictions on test set...")
    poisson_test = poisson_model.predict(test_df)
    poisson_test.columns = ['prob_home', 'prob_draw', 'prob_away']
    poisson_test['Date'] = test_df['Date'].values
    
    catboost_test = load_catboost_predictions(test_df)
    
    merged_test = pd.merge(poisson_test, catboost_test, on='Date')
    y_test = test_df['FTR'].map(result_map).values
    
    # Get ensemble predictions
    ensemble_probs = ensemble.predict_proba(
        poisson_probs=merged_test[['prob_home', 'prob_draw', 'prob_away']],
        ml_probs=merged_test[['prob_home_ml', 'prob_draw_ml', 'prob_away_ml']]
    )
    
    # Calculate metrics
    y_pred_ensemble = ensemble_probs.argmax(axis=1)
    acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
    
    # Brier score
    brier_scores = []
    for i in range(3):
        y_binary = (y_test == i).astype(int)
        brier = brier_score_loss(y_binary, ensemble_probs[:, i])
        brier_scores.append(brier)
    avg_brier_ensemble = np.mean(brier_scores)
    
    # Log loss
    logloss_ensemble = log_loss(y_test, ensemble_probs)
    
    logger.info("\n" + "Ensemble Results:")
    logger.info(f"  Accuracy: {acc_ensemble:.2%}")
    logger.info(f"  Brier Score: {avg_brier_ensemble:.4f}")
    logger.info(f"  Log Loss: {logloss_ensemble:.4f}")
    
    # Compare with individual models
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON: ENSEMBLE VS BASE MODELS")
    logger.info("=" * 70)
    
    # Poisson metrics
    poisson_probs_test = merged_test[['prob_home', 'prob_draw', 'prob_away']].values
    y_pred_poisson = poisson_probs_test.argmax(axis=1)
    acc_poisson = accuracy_score(y_test, y_pred_poisson)
    
    brier_poisson = []
    for i in range(3):
        y_binary = (y_test == i).astype(int)
        brier = brier_score_loss(y_binary, poisson_probs_test[:, i])
        brier_poisson.append(brier)
    avg_brier_poisson = np.mean(brier_poisson)
    logloss_poisson = log_loss(y_test, poisson_probs_test)
    
    # CatBoost metrics
    catboost_probs_test = merged_test[['prob_home_ml', 'prob_draw_ml', 'prob_away_ml']].values
    y_pred_catboost = catboost_probs_test.argmax(axis=1)
    acc_catboost = accuracy_score(y_test, y_pred_catboost)
    
    brier_catboost = []
    for i in range(3):
        y_binary = (y_test == i).astype(int)
        brier = brier_score_loss(y_binary, catboost_probs_test[:, i])
        brier_catboost.append(brier)
    avg_brier_catboost = np.mean(brier_catboost)
    logloss_catboost = log_loss(y_test, catboost_probs_test)
    
    # Print comparison table
    logger.info("\n| Metric      | Poisson (A) | CatBoost (B) | Ensemble | Winner |")
    logger.info("|-------------|-------------|--------------|----------|--------|")
    
    # Accuracy
    best_acc = max(acc_poisson, acc_catboost, acc_ensemble)
    winner_acc = 'A' if best_acc == acc_poisson else ('B' if best_acc == acc_catboost else 'Ens')
    logger.info(f"| Accuracy    | {acc_poisson:11.2%} | {acc_catboost:12.2%} | {acc_ensemble:8.2%} | {winner_acc:6} |")
    
    # Brier
    best_brier = min(avg_brier_poisson, avg_brier_catboost, avg_brier_ensemble)
    winner_brier = 'A' if best_brier == avg_brier_poisson else ('B' if best_brier == avg_brier_catboost else 'Ens')
    logger.info(f"| Brier       | {avg_brier_poisson:11.4f} | {avg_brier_catboost:12.4f} | {avg_brier_ensemble:8.4f} | {winner_brier:6} |")
    
    # Log Loss
    best_logloss = min(logloss_poisson, logloss_catboost, logloss_ensemble)
    winner_logloss = 'A' if best_logloss == logloss_poisson else ('B' if best_logloss == logloss_catboost else 'Ens')
    logger.info(f"| Log Loss    | {logloss_poisson:11.4f} | {logloss_catboost:12.4f} | {logloss_ensemble:8.4f} | {winner_logloss:6} |")
    
    # Save ensemble model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = base_dir / 'models'
    ensemble_file = model_dir / f'ensemble_v1_{timestamp}.pkl'
    ensemble_latest = model_dir / 'ensemble_v1_latest.pkl'
    
    ensemble.save(ensemble_file)
    
    # Create symlink
    if ensemble_latest.exists():
        ensemble_latest.unlink()
    ensemble_latest.symlink_to(ensemble_file.name)
    
    # Save metadata
    metadata = {
        'model': 'Ensemble (Stacking)',
        'version': 'v1',
        'train_date': timestamp,
        'base_models': ['Poisson', 'CatBoost'],
        'meta_model': 'LogisticRegression',
        'calibration': 'Isotonic',
        'val_samples': len(merged_val),
        'test_samples': len(merged_test),
        'metrics': {
            'test': {
                'accuracy': float(acc_ensemble),
                'brier_score': float(avg_brier_ensemble),
                'log_loss': float(logloss_ensemble)
            },
            'comparison': {
                'poisson': {
                    'accuracy': float(acc_poisson),
                    'brier_score': float(avg_brier_poisson),
                    'log_loss': float(logloss_poisson)
                },
                'catboost': {
                    'accuracy': float(acc_catboost),
                    'brier_score': float(avg_brier_catboost),
                    'log_loss': float(logloss_catboost)
                }
            }
        }
    }
    
    meta_file = model_dir / f'ensemble_metadata_{timestamp}.json'
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save test predictions
    test_pred = pd.DataFrame({
        'Date': merged_test['Date'],
        'prob_home_poisson': merged_test['prob_home'],
        'prob_draw_poisson': merged_test['prob_draw'],
        'prob_away_poisson': merged_test['prob_away'],
        'prob_home_catboost': merged_test['prob_home_ml'],
        'prob_draw_catboost': merged_test['prob_draw_ml'],
        'prob_away_catboost': merged_test['prob_away_ml'],
        'prob_home_ensemble': ensemble_probs[:, 2],  # Note: sklearn order is A/D/H
        'prob_draw_ensemble': ensemble_probs[:, 1],
        'prob_away_ensemble': ensemble_probs[:, 0],
        'FTR': test_df['FTR'].values
    })
    
    pred_file = model_dir / f'ensemble_test_predictions_{timestamp}.csv'
    test_pred.to_csv(pred_file, index=False)
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ ENSEMBLE TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Model: {ensemble_file}")
    logger.info(f"Predictions: {pred_file}")
    logger.info(f"Metadata: {meta_file}")
    
    # Summary
    improvement_brier = avg_brier_catboost - avg_brier_ensemble
    improvement_pct = (improvement_brier / avg_brier_catboost) * 100
    
    logger.info(f"\n📊 Ensemble Performance:")
    logger.info(f"  Brier improvement over CatBoost: {improvement_brier:+.4f} ({improvement_pct:+.1f}%)")
    
    if avg_brier_ensemble < min(avg_brier_poisson, avg_brier_catboost):
        logger.info(f"  ✅ Ensemble BEATS both base models!")
    elif avg_brier_ensemble < avg_brier_catboost:
        logger.info(f"  ✅ Ensemble BEATS CatBoost")
    else:
        logger.info(f"  ⚠️  Ensemble does not improve over best base model")


if __name__ == '__main__':
    main()
