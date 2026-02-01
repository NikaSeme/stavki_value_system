#!/usr/bin/env python3
"""
Per-League Market Benchmark Script

Compares market baseline vs model on held-out test set.
Produces:
- Market vs Model metrics table (per league)
- Draw calibration bins
- Data integrity verification

Part 1 & 2 of FIX_INPUTS_PACKAGE
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
from src.models.league_loader import LeagueModelLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR = Path(__file__).parent.parent

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

SPORT_KEY_MAP = {
    'epl': 'soccer_epl',
    'laliga': 'soccer_spain_la_liga',
    'seriea': 'soccer_italy_serie_a',
    'bundesliga': 'soccer_germany_bundesliga',
    'ligue1': 'soccer_france_ligue_one',
    'championship': 'soccer_england_league1',
}


def compute_brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute multi-class Brier score."""
    brier = 0.0
    for c in range(3):
        binary = (y_true == c).astype(float)
        brier += np.mean((y_proba[:, c] - binary) ** 2)
    return brier / 3


def compute_market_probs(df: pd.DataFrame) -> np.ndarray:
    """
    Compute market implied probabilities from odds.
    Uses no-vig probabilities (already in dataset).
    """
    # no_vig columns already normalized
    probs = df[['no_vig_home', 'no_vig_draw', 'no_vig_away']].values
    
    # Ensure valid probabilities
    probs = np.clip(probs, 0.001, 0.999)
    probs = probs / probs.sum(axis=1, keepdims=True)
    
    return probs


def compute_calibration_bins(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    outcome_idx: int,
    n_bins: int = 10
) -> pd.DataFrame:
    """Compute calibration table for one outcome."""
    probs = y_proba[:, outcome_idx]
    actual = (y_true == outcome_idx).astype(float)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= low) & (probs < high)
        
        if mask.sum() > 0:
            rows.append({
                'bin': f"{low:.1f}-{high:.1f}",
                'mean_pred': probs[mask].mean(),
                'mean_actual': actual[mask].mean(),
                'count': int(mask.sum()),
            })
        else:
            rows.append({
                'bin': f"{low:.1f}-{high:.1f}",
                'mean_pred': (low + high) / 2,
                'mean_actual': np.nan,
                'count': 0,
            })
    
    return pd.DataFrame(rows)


def verify_data_integrity(df: pd.DataFrame, league_name: str) -> Dict[str, Any]:
    """
    Verify data integrity for a league.
    
    Checks:
    1. Sorted by datetime
    2. Date ranges
    3. Class distribution
    4. No leakage indicators
    """
    df = df.sort_values('kickoff_time').reset_index(drop=True)
    
    # Check sorting
    dates = pd.to_datetime(df['kickoff_time'])
    is_sorted = dates.is_monotonic_increasing
    
    # Date ranges
    min_date = dates.min()
    max_date = dates.max()
    
    # Class distribution
    label_counts = df['label'].value_counts(normalize=True).sort_index()
    pct_home = label_counts.get(0, 0) * 100
    pct_draw = label_counts.get(1, 0) * 100
    pct_away = label_counts.get(2, 0) * 100
    
    # Check for suspicious columns (post-match data)
    suspicious_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ['fthg', 'ftag', 'ftr', 'result', 'score', 'goal']):
            suspicious_cols.append(col)
    
    # Check ELO values exist and are reasonable
    elo_ok = True
    if 'elo_home' in df.columns and 'elo_away' in df.columns:
        elo_range = (df['elo_home'].min(), df['elo_home'].max())
        elo_ok = elo_range[0] >= 1000 and elo_range[1] <= 2500
    
    return {
        'league': league_name,
        'is_sorted': is_sorted,
        'min_date': str(min_date.date()),
        'max_date': str(max_date.date()),
        'n_samples': len(df),
        'pct_home': pct_home,
        'pct_draw': pct_draw,
        'pct_away': pct_away,
        'suspicious_cols': suspicious_cols,
        'elo_ok': elo_ok,
    }


def evaluate_market_vs_model(
    df: pd.DataFrame,
    league_name: str,
    loader: LeagueModelLoader,
    contract
) -> Dict[str, Any]:
    """
    Compare market baseline vs model on test set.
    """
    # Prepare data
    y_true = df['label'].values
    
    # Market probabilities (from no-vig odds)
    market_probs = compute_market_probs(df)
    
    # Model probabilities
    sport_key = SPORT_KEY_MAP.get(league_name, 'soccer_epl')
    model_probs = loader.predict(df, sport_key=sport_key)
    
    # Metrics
    market_ll = log_loss(y_true, market_probs, labels=[0, 1, 2])
    market_brier = compute_brier(y_true, market_probs)
    market_acc = accuracy_score(y_true, market_probs.argmax(axis=1))
    
    model_ll = log_loss(y_true, model_probs, labels=[0, 1, 2])
    model_brier = compute_brier(y_true, model_probs)
    model_acc = accuracy_score(y_true, model_probs.argmax(axis=1))
    
    # Class distribution on test
    label_counts = pd.Series(y_true).value_counts(normalize=True).sort_index()
    
    # Draw calibration
    draw_cal_market = compute_calibration_bins(y_true, market_probs, 1)
    draw_cal_model = compute_calibration_bins(y_true, model_probs, 1)
    
    return {
        'league': league_name,
        'n_test': len(df),
        'pct_home': label_counts.get(0, 0) * 100,
        'pct_draw': label_counts.get(1, 0) * 100,
        'pct_away': label_counts.get(2, 0) * 100,
        'market_logloss': market_ll,
        'model_logloss': model_ll,
        'diff_logloss': model_ll - market_ll,
        'market_brier': market_brier,
        'model_brier': model_brier,
        'diff_brier': model_brier - market_brier,
        'market_accuracy': market_acc,
        'model_accuracy': model_acc,
        'diff_accuracy': model_acc - market_acc,
        'draw_cal_market': draw_cal_market,
        'draw_cal_model': draw_cal_model,
    }


def print_sample_rows(df: pd.DataFrame, n: int = 10):
    """Print sample rows for verification."""
    sample = df.head(n)[['kickoff_time', 'league', 'home_team', 'away_team', 
                          'odds_home', 'odds_draw', 'odds_away', 
                          'elo_home', 'elo_away', 'label']].copy()
    
    print("\n📋 Sample rows (first 10):")
    print(sample.to_string(index=False))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Market vs Model benchmark per league")
    parser.add_argument('--league', type=str, help="Evaluate single league only")
    parser.add_argument('--output', type=str, default=str(OUTPUT_DIR / "market_benchmark_results.csv"))
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("PER-LEAGUE MARKET BENCHMARK")
    print("=" * 70)
    
    # Load data
    dataset_path = DATA_DIR / "ml_dataset_v2.csv"
    logger.info(f"Loading {dataset_path}")
    df = pd.read_csv(dataset_path)
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    
    # Sort globally
    df = df.sort_values('kickoff_time').reset_index(drop=True)
    
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Sorted: {df['kickoff_time'].is_monotonic_increasing}")
    
    # Load contract and model loader
    contract = load_contract()
    loader = LeagueModelLoader()
    
    # Split by league
    league_col = 'league'
    league_data = {}
    
    for league_code in df[league_col].unique():
        league_name = LEAGUE_MAP.get(league_code, league_code)
        league_df = df[df[league_col] == league_code].copy()
        league_df = league_df.sort_values('kickoff_time').reset_index(drop=True)
        league_data[league_name] = league_df
    
    # Filter if single league requested
    if args.league:
        if args.league in league_data:
            league_data = {args.league: league_data[args.league]}
        else:
            print(f"❌ League {args.league} not found. Available: {list(league_data.keys())}")
            return
    
    # Part 1: Data integrity verification
    print("\n" + "=" * 70)
    print("PART 1: DATA INTEGRITY VERIFICATION")
    print("=" * 70)
    
    integrity_results = []
    
    for league_name, league_df in league_data.items():
        result = verify_data_integrity(league_df, league_name)
        integrity_results.append(result)
        
        print(f"\n📊 {league_name.upper()}")
        print(f"  Sorted: {'✅' if result['is_sorted'] else '❌'}")
        print(f"  Date range: {result['min_date']} to {result['max_date']}")
        print(f"  Samples: {result['n_samples']}")
        print(f"  Class dist: H={result['pct_home']:.1f}% / D={result['pct_draw']:.1f}% / A={result['pct_away']:.1f}%")
        print(f"  Suspicious cols: {result['suspicious_cols'] if result['suspicious_cols'] else '✅ None'}")
        print(f"  ELO valid: {'✅' if result['elo_ok'] else '❌'}")
    
    # Print sample rows from first league
    first_league = list(league_data.values())[0]
    print_sample_rows(first_league)
    
    # Part 2: Market vs Model benchmark
    print("\n" + "=" * 70)
    print("PART 2: MARKET BASELINE VS MODEL")
    print("=" * 70)
    
    benchmark_results = []
    
    for league_name, league_df in league_data.items():
        # Use last 15% as test set (same as training)
        n = len(league_df)
        test_start = int(n * 0.85)
        test_df = league_df.iloc[test_start:].copy()
        
        logger.info(f"\n{league_name}: Test set = {len(test_df)} samples")
        
        result = evaluate_market_vs_model(test_df, league_name, loader, contract)
        benchmark_results.append(result)
    
    # Print summary table
    print("\n" + "=" * 70)
    print("MARKET VS MODEL SUMMARY")
    print("=" * 70)
    
    print(f"\n{'League':<12} {'#Test':>6} {'%H/%D/%A':>12} {'Mkt LL':>8} {'Mdl LL':>8} {'Δ LL':>7} {'Mkt Br':>8} {'Mdl Br':>8} {'Mkt Acc':>8} {'Mdl Acc':>8}")
    print("-" * 100)
    
    for r in benchmark_results:
        dist = f"{r['pct_home']:.0f}/{r['pct_draw']:.0f}/{r['pct_away']:.0f}"
        print(f"{r['league']:<12} {r['n_test']:>6} {dist:>12} {r['market_logloss']:>8.4f} {r['model_logloss']:>8.4f} "
              f"{r['diff_logloss']:>+7.4f} {r['market_brier']:>8.4f} {r['model_brier']:>8.4f} "
              f"{r['market_accuracy']:>8.4f} {r['model_accuracy']:>8.4f}")
    
    # Draw calibration for each league
    print("\n" + "=" * 70)
    print("DRAW CALIBRATION BINS")
    print("=" * 70)
    
    for r in benchmark_results:
        print(f"\n{r['league'].upper()} - Market Draw Calibration:")
        print(r['draw_cal_market'].to_string(index=False))
        
        print(f"\n{r['league'].upper()} - Model Draw Calibration:")
        print(r['draw_cal_model'].to_string(index=False))
    
    # Save results
    summary_df = pd.DataFrame([{
        'league': r['league'],
        'n_test': r['n_test'],
        'pct_home': r['pct_home'],
        'pct_draw': r['pct_draw'],
        'pct_away': r['pct_away'],
        'market_logloss': r['market_logloss'],
        'model_logloss': r['model_logloss'],
        'diff_logloss': r['diff_logloss'],
        'market_brier': r['market_brier'],
        'model_brier': r['model_brier'],
        'diff_brier': r['diff_brier'],
        'market_accuracy': r['market_accuracy'],
        'model_accuracy': r['model_accuracy'],
        'diff_accuracy': r['diff_accuracy'],
    } for r in benchmark_results])
    
    output_path = Path(args.output)
    summary_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved benchmark results to {output_path}")
    
    # Save integrity results
    integrity_df = pd.DataFrame(integrity_results)
    integrity_path = output_path.parent / "data_integrity_results.csv"
    integrity_df.to_csv(integrity_path, index=False)
    print(f"✅ Saved integrity results to {integrity_path}")


if __name__ == "__main__":
    main()
