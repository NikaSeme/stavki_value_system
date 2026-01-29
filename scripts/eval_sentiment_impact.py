"""
Sentiment Impact Evaluation (Task H).

A/B comparison:
- Model A: Without sentiment features
- Model B: With sentiment features

If Brier improvement < 0.5%, sentiment will be removed from pipeline.

Usage:
    python scripts/eval_sentiment_impact.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_setup import get_logger
from sklearn.metrics import brier_score_loss

logger = get_logger(__name__)


def evaluate_model_brier(y_true, y_proba):
    """Calculate average Brier score across classes."""
    brier_scores = []
    for i in range(3):
        y_binary = (y_true == i).astype(int)
        if len(np.unique(y_binary)) > 1:
            brier_scores.append(brier_score_loss(y_binary, y_proba[:, i]))
    return np.mean(brier_scores) if brier_scores else 0.0


def main():
    logger.info("=" * 70)
    logger.info("SENTIMENT A/B TEST (Task H)")
    logger.info("=" * 70)
    
    # Load data
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / 'data' / 'processed' / 'multi_league_features_2021_2024.csv'
    
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Check for sentiment features
    sentiment_cols = [col for col in df.columns if 'Sentiment' in col or 'sentiment' in col]
    
    logger.info(f"Total matches: {len(df)}")
    logger.info(f"Sentiment columns found: {sentiment_cols or 'None'}")
    
    if not sentiment_cols:
        logger.warning("\n⚠️ No sentiment features found in dataset.")
        logger.info("To add sentiment features, run engineer_multi_league_features.py with sentiment support.")
        logger.info("\nA/B Test Result: SKIPPED (no sentiment data available)")
        logger.info("Decision: Cannot evaluate sentiment impact without features.")
        return
    
    # Split data
    n = len(df)
    train_end = int(n * 0.60)
    val_end = int(n * 0.75)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    
    logger.info(f"\nData split:")
    logger.info(f"  Train: {len(train_df)} matches")
    logger.info(f"  Val: {len(val_df)} matches")
    
    # Define feature sets
    base_features = [col for col in df.columns if col not in [
        'Date', 'HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 
        'Season', 'League', 'GoalDiff', 'TotalGoals'
    ] + sentiment_cols]
    
    features_with_sentiment = base_features + sentiment_cols
    
    logger.info(f"\nFeatures (Model A - no sentiment): {len(base_features)}")
    logger.info(f"Features (Model B - with sentiment): {len(features_with_sentiment)}")
    
    # Train simple models for A/B comparison
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    result_map = {'H': 0, 'D': 1, 'A': 2}
    y_train = train_df['FTR'].map(result_map).values
    y_val = val_df['FTR'].map(result_map).values
    
    # Model A: Without sentiment
    logger.info("\nTraining Model A (without sentiment)...")
    X_train_a = train_df[base_features].fillna(0).values
    X_val_a = val_df[base_features].fillna(0).values
    
    scaler_a = StandardScaler()
    X_train_a = scaler_a.fit_transform(X_train_a)
    X_val_a = scaler_a.transform(X_val_a)
    
    model_a = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model_a.fit(X_train_a, y_train)
    proba_a = model_a.predict_proba(X_val_a)
    brier_a = evaluate_model_brier(y_val, proba_a)
    
    logger.info(f"  Brier Score (A): {brier_a:.4f}")
    
    # Model B: With sentiment
    logger.info("\nTraining Model B (with sentiment)...")
    X_train_b = train_df[features_with_sentiment].fillna(0).values
    X_val_b = val_df[features_with_sentiment].fillna(0).values
    
    scaler_b = StandardScaler()
    X_train_b = scaler_b.fit_transform(X_train_b)
    X_val_b = scaler_b.transform(X_val_b)
    
    model_b = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model_b.fit(X_train_b, y_train)
    proba_b = model_b.predict_proba(X_val_b)
    brier_b = evaluate_model_brier(y_val, proba_b)
    
    logger.info(f"  Brier Score (B): {brier_b:.4f}")
    
    # Compare
    improvement = (brier_a - brier_b) / brier_a * 100
    threshold = 0.5  # 0.5% improvement threshold
    
    logger.info("\n" + "=" * 70)
    logger.info("A/B TEST RESULTS")
    logger.info("=" * 70)
    logger.info(f"Model A (no sentiment):   Brier = {brier_a:.4f}")
    logger.info(f"Model B (with sentiment): Brier = {brier_b:.4f}")
    logger.info(f"Improvement: {improvement:+.2f}%")
    
    if improvement >= threshold:
        logger.info(f"\n✅ DECISION: KEEP sentiment features")
        logger.info(f"   Improvement ({improvement:.2f}%) exceeds threshold ({threshold}%)")
        decision = "KEEP"
    else:
        logger.info(f"\n❌ DECISION: REMOVE sentiment features")
        logger.info(f"   Improvement ({improvement:.2f}%) below threshold ({threshold}%)")
        decision = "REMOVE"
    
    # Save results
    results = {
        'brier_without_sentiment': brier_a,
        'brier_with_sentiment': brier_b,
        'improvement_pct': improvement,
        'threshold_pct': threshold,
        'decision': decision,
        'sentiment_features': sentiment_cols
    }
    
    results_file = base_dir / 'models' / 'sentiment_ab_results.json'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
