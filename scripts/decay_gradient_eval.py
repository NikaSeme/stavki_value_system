#!/usr/bin/env python3
"""
Decay Gradient Evaluation Script

Walk-forward CV with quarterly folds to evaluate time decay half-lives.
Produces: decay_gradient_results.csv

Part 3 of FIX_INPUTS_PACKAGE
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import logging
import sys
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, log_loss

from src.models.feature_contract import load_contract

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR = Path(__file__).parent.parent

# Half-lives to test (in days), None = infinite (no decay)
HALF_LIVES = [30, 60, 90, 180, 365, 730, 1095, None]

# League mapping
LEAGUE_MAP = {
    'E0': 'epl',
    'SP1': 'laliga', 
    'I1': 'seriea',
    'D1': 'bundesliga',
    'F1': 'ligue1',
    'ENG2': 'championship',
    'epl': 'epl',
    'laliga': 'laliga',
    'seriea': 'seriea',
    'bundesliga': 'bundesliga',
    'ligue1': 'ligue1',
    'championship': 'championship',
}


@dataclass
class FoldResult:
    """Result from a single fold."""
    fold_idx: int
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    train_size: int
    valid_size: int
    logloss: float
    brier: float
    accuracy: float


def compute_sample_weights(
    dates: pd.Series,
    reference_date: datetime,
    half_life_days: Optional[int]
) -> np.ndarray:
    """Compute exponential decay weights."""
    if half_life_days is None or pd.isna(half_life_days):
        return np.ones(len(dates))
    
    half_life_days = int(half_life_days)
    days_ago = (reference_date - dates).dt.days
    decay_rate = np.log(2) / half_life_days
    weights = np.exp(-decay_rate * days_ago)
    
    return weights.values


def compute_brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute multi-class Brier score."""
    brier = 0.0
    for c in range(3):
        binary = (y_true == c).astype(float)
        brier += np.mean((y_proba[:, c] - binary) ** 2)
    return brier / 3


def generate_quarterly_folds(
    df: pd.DataFrame,
    min_train_samples: int = 200,
    fold_size_days: int = 90
) -> List[Tuple[pd.DataFrame, pd.DataFrame, Dict]]:
    """
    Generate walk-forward quarterly folds.
    
    Returns:
        List of (train_df, valid_df, fold_info) tuples
    """
    df = df.sort_values('kickoff_time').reset_index(drop=True)
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    
    min_date = df['kickoff_time'].min()
    max_date = df['kickoff_time'].max()
    
    folds = []
    
    # Start generating folds after we have enough training data
    # Start validation from 1 year after the first match
    valid_start = min_date + timedelta(days=365)
    
    fold_idx = 0
    while valid_start + timedelta(days=fold_size_days) <= max_date:
        valid_end = valid_start + timedelta(days=fold_size_days)
        
        train_mask = df['kickoff_time'] < valid_start
        valid_mask = (df['kickoff_time'] >= valid_start) & (df['kickoff_time'] < valid_end)
        
        train_df = df[train_mask]
        valid_df = df[valid_mask]
        
        if len(train_df) >= min_train_samples and len(valid_df) >= 10:
            fold_info = {
                'fold_idx': fold_idx,
                'train_start': str(train_df['kickoff_time'].min().date()),
                'train_end': str(train_df['kickoff_time'].max().date()),
                'valid_start': str(valid_df['kickoff_time'].min().date()),
                'valid_end': str(valid_df['kickoff_time'].max().date()),
                'train_size': len(train_df),
                'valid_size': len(valid_df),
            }
            folds.append((train_df.copy(), valid_df.copy(), fold_info))
            fold_idx += 1
        
        valid_start = valid_end
    
    return folds


def train_and_evaluate_fold(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    contract,
    half_life_days: Optional[int]
) -> Dict[str, float]:
    """Train on fold and evaluate."""
    feature_cols = contract.features
    cat_features = list(contract.categorical)
    
    # Prepare data
    X_train = train_df[feature_cols].copy()
    y_train = train_df['label'].values
    X_valid = valid_df[feature_cols].copy()
    y_valid = valid_df['label'].values
    
    # Fill NaN
    for col in contract.numeric:
        if col in X_train.columns:
            X_train[col] = X_train[col].fillna(0.0)
            X_valid[col] = X_valid[col].fillna(0.0)
    
    for col in cat_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].fillna('Unknown').astype(str)
            X_valid[col] = X_valid[col].fillna('Unknown').astype(str)
    
    # Compute weights
    reference_date = train_df['kickoff_time'].max()
    weights = compute_sample_weights(train_df['kickoff_time'], reference_date, half_life_days)
    
    # Cat feature indices
    cat_indices = [X_train.columns.get_loc(c) for c in cat_features if c in X_train.columns]
    
    # Create pools
    train_pool = Pool(X_train, y_train, cat_features=cat_indices, weight=weights)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_indices)
    
    # Train (fast settings for CV)
    params = {
        'iterations': 300,
        'depth': 5,
        'learning_rate': 0.05,
        'l2_leaf_reg': 9,
        'loss_function': 'MultiClass',
        'random_seed': 42,
        'verbose': 0,
        'early_stopping_rounds': 30,
    }
    
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    
    # Evaluate
    y_proba = model.predict_proba(valid_pool)
    
    logloss = log_loss(y_valid, y_proba, labels=[0, 1, 2])
    brier = compute_brier(y_valid, y_proba)
    accuracy = accuracy_score(y_valid, y_proba.argmax(axis=1))
    
    return {
        'logloss': logloss,
        'brier': brier,
        'accuracy': accuracy,
    }


def evaluate_decay_for_league(
    league_name: str,
    folds: List[Tuple[pd.DataFrame, pd.DataFrame, Dict]],
    contract
) -> List[Dict[str, Any]]:
    """
    Evaluate all decay half-lives for a league using walk-forward CV.
    
    IMPORTANT: Same folds are used for ALL half-life settings (apples-to-apples).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATING: {league_name}")
    logger.info(f"{'='*60}")
    
    n_folds = len(folds)
    
    logger.info(f"Using {n_folds} quarterly folds (same for all decay settings)")
    
    if n_folds == 0:
        logger.warning(f"No folds generated for {league_name}")
        return []
    
    # Print fold info ONCE (same for all half-lives)
    for train_df, valid_df, info in folds:
        logger.info(f"  Fold {info['fold_idx']}: Train {info['train_start']} to {info['train_end']} ({info['train_size']}), "
                   f"Valid {info['valid_start']} to {info['valid_end']} ({info['valid_size']})")
    
    results = []
    
    for half_life in HALF_LIVES:
        hl_name = f"{half_life}d" if half_life else "INF"
        
        fold_metrics = []
        
        for train_df, valid_df, fold_info in folds:
            try:
                metrics = train_and_evaluate_fold(train_df, valid_df, contract, half_life)
                fold_metrics.append(metrics)
            except Exception as e:
                logger.warning(f"Fold {fold_info['fold_idx']} failed for {hl_name}: {e}")
        
        if len(fold_metrics) == 0:
            continue
        
        # Aggregate
        logloss_vals = [m['logloss'] for m in fold_metrics]
        brier_vals = [m['brier'] for m in fold_metrics]
        accuracy_vals = [m['accuracy'] for m in fold_metrics]
        
        result = {
            'league': league_name,
            'decay_type': 'half_life',
            'half_life_days': half_life if half_life else 'INF',
            'n_folds': len(fold_metrics),
            'mean_logloss': np.mean(logloss_vals),
            'std_logloss': np.std(logloss_vals),
            'mean_brier': np.mean(brier_vals),
            'std_brier': np.std(brier_vals),
            'mean_accuracy': np.mean(accuracy_vals),
            'std_accuracy': np.std(accuracy_vals),
            'chosen_best_flag': False,  # Will be set later
        }
        
        results.append(result)
        
        logger.info(f"  {hl_name:>6s}: LogLoss={result['mean_logloss']:.4f}±{result['std_logloss']:.4f}, "
                   f"Brier={result['mean_brier']:.4f}±{result['std_brier']:.4f}")
    
    # Mark best by mean LogLoss
    if results:
        best_idx = np.argmin([r['mean_logloss'] for r in results])
        results[best_idx]['chosen_best_flag'] = True
        logger.info(f"\n  ✅ Best: {results[best_idx]['half_life_days']} (LogLoss={results[best_idx]['mean_logloss']:.4f})")
    
    return results


def save_fold_info(folds_by_league: Dict[str, List], output_path: Path):
    """Save detailed fold information."""
    rows = []
    for league, folds in folds_by_league.items():
        for train_df, valid_df, info in folds:
            rows.append({
                'league': league,
                'fold_idx': info['fold_idx'],
                'train_start': info['train_start'],
                'train_end': info['train_end'],
                'valid_start': info['valid_start'],
                'valid_end': info['valid_end'],
                'train_size': info['train_size'],
                'valid_size': info['valid_size'],
            })
    
    fold_df = pd.DataFrame(rows)
    fold_path = output_path.parent / "decay_gradient_folds.csv"
    fold_df.to_csv(fold_path, index=False)
    logger.info(f"Saved fold info to {fold_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Decay gradient evaluation with walk-forward CV")
    parser.add_argument('--league', type=str, help="Evaluate single league only")
    parser.add_argument('--output', type=str, default=str(OUTPUT_DIR / "decay_gradient_results.csv"))
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("DECAY GRADIENT EVALUATION")
    print("Walk-forward CV with quarterly folds")
    print("=" * 70)
    
    # Load data
    dataset_path = DATA_DIR / "ml_dataset_v2.csv"
    logger.info(f"Loading {dataset_path}")
    df = pd.read_csv(dataset_path)
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Date range: {df['kickoff_time'].min()} to {df['kickoff_time'].max()}")
    
    # Load contract
    contract = load_contract()
    
    # Split by league
    league_col = 'league'
    league_data = {}
    
    for league_code in df[league_col].unique():
        league_name = LEAGUE_MAP.get(league_code, league_code)
        league_data[league_name] = df[df[league_col] == league_code].copy()
    
    # Filter if single league requested
    if args.league:
        if args.league in league_data:
            league_data = {args.league: league_data[args.league]}
        else:
            print(f"❌ League {args.league} not found. Available: {list(league_data.keys())}")
            return
    
    # Evaluate each league
    all_results = []
    folds_by_league = {}
    
    for league_name, league_df in league_data.items():
        # Generate folds ONCE per league (used for all half-life comparisons)
        folds = generate_quarterly_folds(league_df)
        folds_by_league[league_name] = folds
        
        # Evaluate all half-lives using the SAME folds
        results = evaluate_decay_for_league(league_name, folds, contract)
        all_results.extend(results)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    output_path = Path(args.output)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved results to {output_path}")
    
    # Save fold info
    save_fold_info(folds_by_league, output_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: BEST DECAY PER LEAGUE")
    print("=" * 70)
    
    best_per_league = results_df[results_df['chosen_best_flag'] == True]
    
    print(f"\n{'League':<15} {'Best Decay':>12} {'LogLoss':>15} {'Brier':>15} {'Folds':>6}")
    print("-" * 70)
    
    for _, row in best_per_league.iterrows():
        ll_str = f"{row['mean_logloss']:.4f}±{row['std_logloss']:.4f}"
        br_str = f"{row['mean_brier']:.4f}±{row['std_brier']:.4f}"
        print(f"{row['league']:<15} {str(row['half_life_days']):>12} {ll_str:>15} {br_str:>15} {row['n_folds']:>6}")


if __name__ == "__main__":
    main()
