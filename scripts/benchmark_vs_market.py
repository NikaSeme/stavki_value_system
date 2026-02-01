#!/usr/bin/env python3
"""
Benchmark vs Market (Task 8)

Compares ML model predictions against market-implied probabilities.
The market baseline uses no-vig odds as the "market's probability estimate".

Success criteria: Model LogLoss < Market LogLoss
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import json
import joblib
import logging

from catboost import CatBoostClassifier
from sklearn.metrics import log_loss

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.snapshot_config import FEATURE_ORDER, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.models.feature_contract import validate_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
DATASET_PATH = DATA_DIR / "snapshot_dataset.csv"


def load_resources():
    """Load dataset, model, and calibrator."""
    # Dataset
    df = pd.read_csv(DATASET_PATH)
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    
    # Model
    model = CatBoostClassifier()
    model.load_model(str(MODELS_DIR / "catboost_snapshot.cbm"))
    
    # Calibrator (optional)
    calibrator_path = MODELS_DIR / "calibrator.pkl"
    if calibrator_path.exists():
        # Need to import the class before unpickling
        from src.models.snapshot_calibrator import MultiClassCalibrator
        calibrator = joblib.load(calibrator_path)
    else:
        calibrator = None
    
    return df, model, calibrator


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Extract feature matrix and labels."""
    X = df[FEATURE_ORDER].copy()
    y = df['label'].values
    
    # Fill NaN
    for col in NUMERIC_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna(0.0)
    
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna('Unknown').astype(str)
    
    return X, y


def get_market_probs(df: pd.DataFrame) -> np.ndarray:
    """
    Get market-implied probabilities (no-vig).
    These represent the bookmakers' probability estimates.
    """
    probs = df[['no_vig_home', 'no_vig_draw', 'no_vig_away']].values
    
    # Ensure sums to 1
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / np.maximum(row_sums, 1e-10)
    
    return probs


def compute_brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute multi-class Brier score."""
    n_classes = y_proba.shape[1]
    y_onehot = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true):
        if 0 <= label < n_classes:
            y_onehot[i, label] = 1
    return np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1))


def benchmark_overall(
    df: pd.DataFrame,
    model: CatBoostClassifier,
    calibrator=None,
) -> Dict:
    """Compute overall benchmark metrics."""
    X, y = prepare_features(df)
    
    # Model predictions
    y_model_raw = model.predict_proba(X)
    
    if calibrator:
        y_model = calibrator.predict_proba(y_model_raw)
    else:
        y_model = y_model_raw
    
    # Market predictions
    y_market = get_market_probs(df)
    
    # Metrics
    market_ll = log_loss(y, y_market, labels=[0, 1, 2])
    model_ll = log_loss(y, y_model, labels=[0, 1, 2])
    
    market_brier = compute_brier(y, y_market)
    model_brier = compute_brier(y, y_model)
    
    return {
        'n_samples': len(y),
        'market_logloss': market_ll,
        'model_logloss': model_ll,
        'logloss_diff': market_ll - model_ll,
        'logloss_pct_diff': (market_ll - model_ll) / market_ll * 100,
        'market_brier': market_brier,
        'model_brier': model_brier,
        'brier_diff': market_brier - model_brier,
        'model_beats_market': model_ll < market_ll,
    }


def benchmark_by_league(
    df: pd.DataFrame,
    model: CatBoostClassifier,
    calibrator=None,
) -> pd.DataFrame:
    """Compute benchmark metrics per league."""
    results = []
    
    for league in df['league'].unique():
        league_df = df[df['league'] == league]
        
        if len(league_df) < 50:
            continue
        
        metrics = benchmark_overall(league_df, model, calibrator)
        metrics['league'] = league
        results.append(metrics)
    
    return pd.DataFrame(results).sort_values('logloss_diff', ascending=False)


def benchmark_by_horizon(
    df: pd.DataFrame,
    model: CatBoostClassifier,
    calibrator=None,
) -> pd.DataFrame:
    """Compute benchmark metrics per horizon."""
    results = []
    
    for horizon in df['horizon_hours'].unique():
        horizon_df = df[df['horizon_hours'] == horizon]
        
        if len(horizon_df) < 50:
            continue
        
        metrics = benchmark_overall(horizon_df, model, calibrator)
        metrics['horizon'] = horizon
        results.append(metrics)
    
    return pd.DataFrame(results).sort_values('horizon')


def print_benchmark_table(
    df: pd.DataFrame,
    group_col: str = 'league',
    title: str = "BENCHMARK BY LEAGUE"
):
    """Print formatted benchmark table."""
    print(f"\n{'=' * 80}")
    print(title)
    print('=' * 80)
    
    # Header
    print(f"{'Group':<15} {'N':>6} {'Market LL':>10} {'Model LL':>10} {'Δ LL':>8} {'%':>7} {'Beats?':>7}")
    print("-" * 80)
    
    # Rows
    for _, row in df.iterrows():
        group = str(row[group_col])[:15]
        n = row['n_samples']
        mkt_ll = row['market_logloss']
        mod_ll = row['model_logloss']
        diff = row['logloss_diff']
        pct = row.get('logloss_pct_diff', diff / mkt_ll * 100 if mkt_ll > 0 else 0)
        beats = "✅ YES" if row['model_beats_market'] else "❌ NO"
        
        print(f"{group:<15} {n:>6} {mkt_ll:>10.4f} {mod_ll:>10.4f} {diff:>+8.4f} {pct:>+6.2f}% {beats:>7}")
    
    # Summary
    print("-" * 80)
    total = df['n_samples'].sum()
    wins = df['model_beats_market'].sum()
    print(f"{'TOTAL':<15} {total:>6}")
    print(f"Model beats market in {wins}/{len(df)} groups ({wins/len(df)*100:.1f}%)")


def save_benchmark_report(
    overall: Dict,
    by_league: pd.DataFrame,
    output_path: Path,
):
    """Save benchmark report as markdown."""
    with open(output_path, 'w') as f:
        f.write("# Market Benchmark Results\n\n")
        f.write(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        # Overall
        f.write("## Overall Results\n\n")
        f.write(f"| Metric | Market | Model | Difference |\n")
        f.write(f"|--------|--------|-------|------------|\n")
        f.write(f"| LogLoss | {overall['market_logloss']:.4f} | {overall['model_logloss']:.4f} | {overall['logloss_diff']:+.4f} |\n")
        f.write(f"| Brier | {overall['market_brier']:.4f} | {overall['model_brier']:.4f} | {overall['brier_diff']:+.4f} |\n")
        f.write(f"| N Samples | {overall['n_samples']} | - | - |\n\n")
        
        verdict = "✅ **Model beats market**" if overall['model_beats_market'] else "❌ **Market wins**"
        f.write(f"**Verdict**: {verdict}\n\n")
        
        # By league
        f.write("## Results by League\n\n")
        f.write("| League | N | Market LL | Model LL | Δ LL | % | Beats? |\n")
        f.write("|--------|---|-----------|----------|------|---|--------|\n")
        
        for _, row in by_league.iterrows():
            beats = "✅" if row['model_beats_market'] else "❌"
            f.write(
                f"| {row['league']} | {row['n_samples']} | "
                f"{row['market_logloss']:.4f} | {row['model_logloss']:.4f} | "
                f"{row['logloss_diff']:+.4f} | {row['logloss_pct_diff']:+.2f}% | {beats} |\n"
            )
        
        f.write("\n")
    
    logger.info(f"Saved benchmark report to {output_path}")


def main():
    """Run benchmark comparison."""
    print("\n" + "=" * 60)
    print("BENCHMARK: MODEL vs MARKET")
    print("=" * 60)
    
    # Load resources
    df, model, calibrator = load_resources()
    
    # Filter to test set only
    test_df = df[df['kickoff_time'] >= "2024-01-31"]
    logger.info(f"Test set: {len(test_df)} samples")
    
    if len(test_df) == 0:
        print("❌ No test data available")
        return
    
    # Overall benchmark
    print("\n📊 Computing overall metrics...")
    overall = benchmark_overall(test_df, model, calibrator)
    
    print(f"\n{'=' * 50}")
    print("OVERALL RESULTS")
    print('=' * 50)
    print(f"Samples:       {overall['n_samples']:,}")
    print(f"Market LL:     {overall['market_logloss']:.4f}")
    print(f"Model LL:      {overall['model_logloss']:.4f}")
    print(f"Difference:    {overall['logloss_diff']:+.4f} ({overall['logloss_pct_diff']:+.2f}%)")
    print(f"Market Brier:  {overall['market_brier']:.4f}")
    print(f"Model Brier:   {overall['model_brier']:.4f}")
    
    if overall['model_beats_market']:
        print("\n✅ MODEL BEATS MARKET!")
    else:
        print("\n❌ Market wins (model needs improvement)")
    
    # By league
    print("\n📊 Computing per-league metrics...")
    by_league = benchmark_by_league(test_df, model, calibrator)
    print_benchmark_table(by_league, 'league', "BENCHMARK BY LEAGUE")
    
    # Save report
    report_path = MODELS_DIR / "benchmark_report.md"
    save_benchmark_report(overall, by_league, report_path)
    
    print(f"\n✅ Benchmark complete. Report: {report_path}")


if __name__ == "__main__":
    main()
