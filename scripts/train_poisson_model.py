"""
Train Poisson model (Model A) for ensemble.

Calculates team attack/defense strengths from historical data
and generates probability predictions for 3-way outcomes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import pickle
import sys
import argparse
from collections import defaultdict
from scipy.stats import poisson
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_setup import get_logger

logger = get_logger(__name__)


from src.models.poisson_model import PoissonMatchPredictor

# ... (Removed local class definition)


def tune_decay_rate(train_df, val_df, decay_rates=None):
    """
    Grid search to find optimal time_decay_rate (Task G).
    
    Args:
        train_df: Training data
        val_df: Validation data for evaluation
        decay_rates: List of rates to try (default: [0.001, 0.002, 0.003, 0.005, 0.01])
        
    Returns:
        (best_rate, best_brier, results_dict)
    """
    if decay_rates is None:
        decay_rates = [0.001, 0.002, 0.003, 0.005, 0.01]
    
    result_map = {'H': 0, 'D': 1, 'A': 2}
    results = {}
    best_rate, best_brier = None, float('inf')
    
    logger.info(f"Tuning time_decay_rate over {decay_rates}...")
    
    for rate in decay_rates:
        # Train with this rate
        model = PoissonMatchPredictor(home_advantage=0.15, time_decay_rate=rate)
        model.fit(train_df)
        
        # Evaluate on validation set
        preds = model.predict(val_df)
        y_true = val_df['FTR'].map(result_map).values
        
        # Calculate Brier score
        probs_array = preds[['prob_home', 'prob_draw', 'prob_away']].values
        brier_scores = []
        for i in range(3):
            y_binary = (y_true == i).astype(int)
            if len(np.unique(y_binary)) > 1:
                brier_scores.append(brier_score_loss(y_binary, probs_array[:, i]))
        avg_brier = np.mean(brier_scores) if brier_scores else 0.0
        
        results[rate] = avg_brier
        logger.info(f"  rate={rate:.4f}: Brier={avg_brier:.4f}")
        
        if avg_brier < best_brier:
            best_rate, best_brier = rate, avg_brier
    
    logger.info(f"\n✓ Best time_decay_rate: {best_rate:.4f} (Brier={best_brier:.4f})")
    return best_rate, best_brier, results


def main():
    """Train Poisson model and evaluate."""
    # Parse arguments (Task G)
    parser = argparse.ArgumentParser(description='Train Poisson model')
    parser.add_argument('--tune-decay', action='store_true',
                        help='Tune time_decay_rate via grid search')
    parser.add_argument('--decay-rate', type=float, default=0.003,
                        help='Time decay rate (default: 0.003)')
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("POISSON MODEL TRAINING (MODEL A)")
    logger.info("=" * 70)
    
    # Load data
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / 'data' / 'processed' / 'multi_league_features_2021_2024.csv'
    
    logger.info(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    
    # Robust date parsing (matches the safety in fit() method)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    invalid_dates = df['Date'].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"Found {invalid_dates} rows with invalid dates in CSV - dropping them")
        df = df.dropna(subset=['Date'])
    
    df = df.sort_values('Date')
    
    logger.info(f"Total matches: {len(df)}")
    logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Same split as CatBoost for fair comparison
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"\nSplit:")
    logger.info(f"  Train: {len(train_df)} matches ({train_df['Date'].min()} to {train_df['Date'].max()})")
    logger.info(f"  Val:   {len(val_df)} matches ({val_df['Date'].min()} to {val_df['Date'].max()})")
    logger.info(f"  Test:  {len(test_df)} matches ({test_df['Date'].min()} to {test_df['Date'].max()})")
    
    # Task G: Optionally tune time_decay_rate
    if args.tune_decay:
        best_rate, best_brier, tune_results = tune_decay_rate(train_df, val_df)
        decay_rate = best_rate
        logger.info(f"Using tuned decay_rate: {decay_rate}")
    else:
        decay_rate = args.decay_rate
        logger.info(f"Using specified decay_rate: {decay_rate}")
    
    # Train model with time decay
    logger.info("\nTraining Poisson model...")
    model = PoissonMatchPredictor(home_advantage=0.15, time_decay_rate=decay_rate)
    model.fit(train_df)
    
    # Evaluate on all sets
    result_map = {'H': 0, 'D': 1, 'A': 2}
    
    for name, subset in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        logger.info(f"\n{name} Set Evaluation:")
        
        # Predict
        pred_df = model.predict(subset)
        probs = pred_df[['prob_home', 'prob_draw', 'prob_away']].values
        
        # True labels
        y_true = subset['FTR'].map(result_map).values
        
        # Metrics
        y_pred = probs.argmax(axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Brier score (average over classes)
        brier_scores = []
        for i in range(3):
            y_binary = (y_true == i).astype(int)
            brier = brier_score_loss(y_binary, probs[:, i])
            brier_scores.append(brier)
        avg_brier = np.mean(brier_scores)
        
        # Log loss
        logloss = log_loss(y_true, probs)
        
        logger.info(f"  Accuracy: {accuracy:.2%}")
        logger.info(f"  Brier Score: {avg_brier:.4f}")
        logger.info(f"  Log Loss: {logloss:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = base_dir / 'models'
    model_file = model_dir / f'poisson_v1_{timestamp}.pkl'
    model_latest = model_dir / 'poisson_v1_latest.pkl'
    
    model.save(model_file)
    
    # Create symlink for latest
    if model_latest.exists():
        model_latest.unlink()
    model_latest.symlink_to(model_file.name)
    
    # Save test predictions for ensemble training
    test_pred = model.predict(test_df)
    test_pred['Date'] = test_df['Date'].values
    test_pred['HomeTeam'] = test_df['HomeTeam'].values
    test_pred['AwayTeam'] = test_df['AwayTeam'].values
    test_pred['FTR'] = test_df['FTR'].values
    
    pred_file = model_dir / f'poisson_test_predictions_{timestamp}.csv'
    test_pred.to_csv(pred_file, index=False)
    logger.info(f"Test predictions saved to {pred_file}")
    
    # Save metadata
    metadata = {
        'model': 'Poisson',
        'version': 'v1',
        'train_date': timestamp,
        'home_advantage': model.home_advantage,
        'league_avg_goals': model.league_avg_goals,
        'num_teams': len(model.team_attack),
        'train_matches': len(train_df),
        'val_matches': len(val_df),
        'test_matches': len(test_df),
        'metrics': {
            'test': {
                'accuracy': float(accuracy),
                'brier_score': float(avg_brier),
                'log_loss': float(logloss)
            }
        }
    }
    
    meta_file = model_dir / f'poisson_metadata_{timestamp}.json'
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\n{'=' * 70}")
    logger.info("✅ POISSON MODEL TRAINING COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"Model: {model_file}")
    logger.info(f"Predictions: {pred_file}")
    logger.info(f"Metadata: {meta_file}")


if __name__ == '__main__':
    main()
