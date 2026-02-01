#!/usr/bin/env python3
"""
Per-League CatBoost Training with Time Decay Optimization

Phases:
1. Split data by league
2. Optimize time decay per league (walk-forward validation)
3. Train final models with optimal decay
4. Evaluate and report

Usage:
    python scripts/train_per_league.py
    python scripts/train_per_league.py --league soccer_epl  # Single league
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import logging
import sys

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

# League mapping
LEAGUE_MAP = {
    'E0': 'soccer_epl',
    'SP1': 'soccer_spain_la_liga', 
    'I1': 'soccer_italy_serie_a',
    'D1': 'soccer_germany_bundesliga',
    'F1': 'soccer_france_ligue_one',
}

# Decay half-lives to test (in days)
DECAY_HALF_LIVES = [90, 180, 365, 730, 1095, None]  # None = no decay (infinite)


def compute_sample_weights(
    dates: pd.Series,
    reference_date: datetime,
    half_life_days: Optional[int]
) -> np.ndarray:
    """
    Compute exponential decay weights based on time.
    
    Args:
        dates: Series of match dates
        reference_date: Most recent date (weight=1.0)
        half_life_days: Days for weight to halve. None = no decay.
    
    Returns:
        Array of weights (0.0 to 1.0)
    """
    if half_life_days is None or pd.isna(half_life_days):
        return np.ones(len(dates))
    
    half_life_days = int(half_life_days)  # Ensure int
    
    days_ago = (reference_date - dates).dt.days
    # w = exp(-ln(2) * days_ago / half_life)
    decay_rate = np.log(2) / half_life_days
    weights = np.exp(-decay_rate * days_ago)
    
    return weights.values


def simulate_value_betting_roi(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    odds: np.ndarray,
    edge_threshold: float = 0.05,
    kelly_fraction: float = 0.25
) -> Dict[str, float]:
    """
    Simulate value betting returns.
    
    Args:
        y_true: Actual outcomes (0=H, 1=D, 2=A)
        y_proba: Predicted probabilities (n, 3)
        odds: Actual odds for each outcome (n, 3)
        edge_threshold: Minimum edge to bet
        kelly_fraction: Fraction of Kelly to use
    
    Returns:
        Dict with ROI, num_bets, profit
    """
    bankroll = 1000.0
    initial_bankroll = bankroll
    num_bets = 0
    
    for i in range(len(y_true)):
        for outcome in range(3):
            p_model = y_proba[i, outcome]
            o = odds[i, outcome]
            
            if o <= 1.0:
                continue
            
            p_implied = 1.0 / o
            edge = p_model - p_implied
            
            if edge > edge_threshold:
                # Kelly bet
                kelly = (p_model * o - 1) / (o - 1)
                stake = bankroll * kelly * kelly_fraction
                stake = min(stake, bankroll * 0.05)  # Max 5% of bankroll
                stake = max(stake, 0)
                
                if stake > 0:
                    num_bets += 1
                    if y_true[i] == outcome:
                        bankroll += stake * (o - 1)
                    else:
                        bankroll -= stake
    
    profit = bankroll - initial_bankroll
    roi = profit / initial_bankroll if initial_bankroll > 0 else 0
    
    return {
        'roi': roi,
        'num_bets': num_bets,
        'profit': profit,
        'final_bankroll': bankroll
    }


def train_and_evaluate(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    contract,
    sample_weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Train CatBoost and evaluate on validation/test.
    """
    feature_cols = contract.features
    cat_features = list(contract.categorical)
    
    # Prepare data
    X_train = train_df[feature_cols].copy()
    y_train = train_df['label'].values
    X_valid = valid_df[feature_cols].copy()
    y_valid = valid_df['label'].values
    X_test = test_df[feature_cols].copy()
    y_test = test_df['label'].values
    
    # Fill NaN
    for col in contract.numeric:
        if col in X_train.columns:
            X_train[col] = X_train[col].fillna(0.0)
            X_valid[col] = X_valid[col].fillna(0.0)
            X_test[col] = X_test[col].fillna(0.0)
    
    for col in cat_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].fillna('Unknown').astype(str)
            X_valid[col] = X_valid[col].fillna('Unknown').astype(str)
            X_test[col] = X_test[col].fillna('Unknown').astype(str)
    
    # Cat feature indices
    cat_indices = [X_train.columns.get_loc(c) for c in cat_features if c in X_train.columns]
    
    # Create pools
    train_pool = Pool(X_train, y_train, cat_features=cat_indices, weight=sample_weights)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_indices)
    test_pool = Pool(X_test, y_test, cat_features=cat_indices)
    
    # Train
    params = {
        'iterations': 500,
        'depth': 5,
        'learning_rate': 0.03,
        'l2_leaf_reg': 9,
        'loss_function': 'MultiClass',
        'eval_metric': 'MultiClass',
        'random_seed': 42,
        'verbose': 0,
        'early_stopping_rounds': 50,
        'task_type': 'CPU',
    }
    
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    
    # Evaluate
    y_valid_proba = model.predict_proba(valid_pool)
    y_test_proba = model.predict_proba(test_pool)
    
    valid_acc = accuracy_score(y_valid, y_valid_proba.argmax(axis=1))
    valid_ll = log_loss(y_valid, y_valid_proba, labels=[0, 1, 2])
    
    test_acc = accuracy_score(y_test, y_test_proba.argmax(axis=1))
    test_ll = log_loss(y_test, y_test_proba, labels=[0, 1, 2])
    
    # Simulate ROI (need odds from data)
    # For now, use implied from no_vig probabilities as proxy
    if 'odds_home' in test_df.columns:
        test_odds = test_df[['odds_home', 'odds_draw', 'odds_away']].values
        roi_result = simulate_value_betting_roi(y_test, y_test_proba, test_odds)
    else:
        roi_result = {'roi': 0, 'num_bets': 0}
    
    return {
        'model': model,
        'valid_accuracy': valid_acc,
        'valid_logloss': valid_ll,
        'test_accuracy': test_acc,
        'test_logloss': test_ll,
        'test_roi': roi_result['roi'],
        'test_num_bets': roi_result['num_bets'],
    }


def optimize_decay_for_league(
    df: pd.DataFrame,
    league_name: str,
    contract
) -> Dict[str, Any]:
    """
    Find optimal time decay for a league using walk-forward validation.
    """
    # Sort chronologically
    df = df.sort_values('kickoff_time').reset_index(drop=True)
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    
    n = len(df)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    
    train = df.iloc[:train_end]
    valid = df.iloc[train_end:valid_end]
    test = df.iloc[valid_end:]
    
    reference_date = train['kickoff_time'].max()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"OPTIMIZING: {league_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Train: {len(train)} | Valid: {len(valid)} | Test: {len(test)}")
    
    results = []
    
    for half_life in DECAY_HALF_LIVES:
        hl_name = f"{half_life}d" if half_life else "∞"
        
        # Compute weights
        weights = compute_sample_weights(train['kickoff_time'], reference_date, half_life)
        
        # Train and evaluate
        metrics = train_and_evaluate(train, valid, test, contract, weights)
        
        results.append({
            'half_life': half_life,
            'half_life_name': hl_name,
            'valid_logloss': metrics['valid_logloss'],
            'valid_accuracy': metrics['valid_accuracy'],
            'test_logloss': metrics['test_logloss'],
            'test_accuracy': metrics['test_accuracy'],
            'test_roi': metrics['test_roi'],
        })
        
        logger.info(f"  {hl_name:>6s}: Valid LL={metrics['valid_logloss']:.4f}, "
                   f"Test Acc={metrics['test_accuracy']:.4f}, ROI={metrics['test_roi']:+.2%}")
    
    # Select best by validation LogLoss (lower is better)
    results_df = pd.DataFrame(results)
    best_idx = results_df['valid_logloss'].idxmin()
    best = results_df.iloc[best_idx]
    
    logger.info(f"\n  ✅ Best decay: {best['half_life_name']} (Valid LL={best['valid_logloss']:.4f})")
    
    return {
        'league': league_name,
        'optimal_half_life': best['half_life'],
        'optimal_half_life_name': best['half_life_name'],
        'all_results': results,
        'train_size': len(train),
        'valid_size': len(valid),
        'test_size': len(test),
    }


def train_final_model(
    df: pd.DataFrame,
    league_name: str,
    optimal_half_life: Optional[int],
    contract
) -> Dict[str, Any]:
    """
    Train final model with optimal decay.
    """
    df = df.sort_values('kickoff_time').reset_index(drop=True)
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    
    n = len(df)
    train_end = int(n * 0.85)  # Use more data for final model
    
    train = df.iloc[:train_end]
    test = df.iloc[train_end:]
    
    # For final training, use train as both train and valid (we already selected hyperparams)
    valid_split = int(len(train) * 0.85)
    train_final = train.iloc[:valid_split]
    valid_final = train.iloc[valid_split:]
    
    reference_date = train_final['kickoff_time'].max()
    weights = compute_sample_weights(train_final['kickoff_time'], reference_date, optimal_half_life)
    
    metrics = train_and_evaluate(train_final, valid_final, test, contract, weights)
    
    return {
        'model': metrics['model'],
        'test_accuracy': metrics['test_accuracy'],
        'test_logloss': metrics['test_logloss'],
        'test_roi': metrics['test_roi'],
        'test_num_bets': metrics['test_num_bets'],
        'train_size': len(train_final),
        'test_size': len(test),
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train per-league CatBoost models")
    parser.add_argument('--league', type=str, help="Train single league only")
    parser.add_argument('--skip-decay', action='store_true', help="Skip decay optimization")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("PER-LEAGUE CATBOOST TRAINING")
    print("=" * 70)
    
    # Load data
    dataset_path = DATA_DIR / "ml_dataset_v2.csv"
    logger.info(f"Loading {dataset_path}")
    df = pd.read_csv(dataset_path)
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    
    logger.info(f"Total samples: {len(df)}")
    
    # Load contract
    contract = load_contract()
    logger.info(f"Feature contract: {contract.feature_count} features")
    
    # Split by league
    league_col = 'league'
    leagues = df[league_col].unique()
    
    print(f"\n📊 League Distribution:")
    for league in sorted(leagues):
        count = len(df[df[league_col] == league])
        print(f"  {league}: {count} samples")
    
    # Map leagues
    league_data = {}
    for div_code in leagues:
        league_key = LEAGUE_MAP.get(div_code, div_code)
        league_data[league_key] = df[df[league_col] == div_code].copy()
    
    # Filter if single league requested
    if args.league:
        if args.league in league_data:
            league_data = {args.league: league_data[args.league]}
        else:
            print(f"❌ League {args.league} not found. Available: {list(league_data.keys())}")
            return
    
    # Phase 2: Optimize decay for each league
    decay_config = {}
    
    if not args.skip_decay:
        print("\n" + "=" * 70)
        print("PHASE 2: TIME DECAY OPTIMIZATION")
        print("=" * 70)
        
        for league_key, league_df in league_data.items():
            result = optimize_decay_for_league(league_df, league_key, contract)
            decay_config[league_key] = {
                'optimal_half_life': result['optimal_half_life'],
                'train_size': result['train_size'],
            }
        
        # Save decay config
        config_path = MODELS_DIR / "league_decay_config.json"
        with open(config_path, 'w') as f:
            json.dump(decay_config, f, indent=2)
        logger.info(f"Saved decay config to {config_path}")
    else:
        # Load existing config or use default
        config_path = MODELS_DIR / "league_decay_config.json"
        if config_path.exists():
            with open(config_path) as f:
                decay_config = json.load(f)
        else:
            decay_config = {k: {'optimal_half_life': 365} for k in league_data.keys()}
    
    # Phase 3: Train final models
    print("\n" + "=" * 70)
    print("PHASE 3: TRAINING FINAL MODELS")
    print("=" * 70)
    
    final_results = {}
    
    for league_key, league_df in league_data.items():
        half_life = decay_config.get(league_key, {}).get('optimal_half_life', 365)
        hl_name = f"{half_life}d" if half_life else "∞"
        
        print(f"\n🚀 Training {league_key} (decay={hl_name})...")
        
        result = train_final_model(league_df, league_key, half_life, contract)
        
        # Save model
        model_path = MODELS_DIR / f"catboost_{league_key.replace('soccer_', '')}.cbm"
        result['model'].save_model(str(model_path))
        logger.info(f"  Saved model to {model_path}")
        
        final_results[league_key] = {
            'accuracy': result['test_accuracy'],
            'logloss': result['test_logloss'],
            'roi': result['test_roi'],
            'num_bets': result['test_num_bets'],
            'decay_half_life': half_life,
        }
        
        print(f"  ✅ Acc={result['test_accuracy']:.4f}, LL={result['test_logloss']:.4f}, ROI={result['test_roi']:+.2%}")
    
    # Phase 4: Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'League':<30} {'Decay':>8} {'Accuracy':>10} {'LogLoss':>10} {'ROI':>10}")
    print("-" * 70)
    
    for league, metrics in final_results.items():
        hl = f"{metrics['decay_half_life']}d" if metrics['decay_half_life'] else "∞"
        print(f"{league:<30} {hl:>8} {metrics['accuracy']:>10.4f} {metrics['logloss']:>10.4f} {metrics['roi']:>+10.2%}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'leagues': final_results,
        'decay_config': decay_config,
        'feature_count': contract.feature_count,
    }
    
    meta_path = MODELS_DIR / "per_league_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ All models saved to {MODELS_DIR}")
    print(f"✅ Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
