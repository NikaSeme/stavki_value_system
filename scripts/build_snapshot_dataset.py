#!/usr/bin/env python3
"""
Build Snapshot Training Dataset (Task 3)

Creates training dataset from historical matches using odds available at match time.
For historical data without snapshots, uses closing odds as T-0h proxy.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.snapshot_features import (
    SnapshotFeatureBuilder, 
    build_market_features,
    make_features
)
from src.config.snapshot_config import FEATURE_ORDER, FEATURE_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_PATH = DATA_DIR / "snapshot_dataset.parquet"

# League mapping (to normalize names)
LEAGUE_MAP = {
    'epl': 'EPL',
    'championship': 'Championship', 
    'bundesliga': 'Bundesliga',
    'laliga': 'LaLiga',
    'seriea': 'SerieA',
    'ligue1': 'Ligue1',
}


def load_historical_data() -> pd.DataFrame:
    """Load the multi-league historical dataset."""
    csv_path = PROCESSED_DIR / "multi_league_features_2021_2024.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} matches from {csv_path}")
    return df


def extract_odds_columns(row: pd.Series) -> Tuple[float, float, float, List, List, List]:
    """
    Extract best odds and all bookmaker odds from a row.
    
    Returns:
        best_home, best_draw, best_away, all_home, all_draw, all_away
    """
    # Odds columns in dataset
    home_cols = ['B365H', 'PSH', 'MaxH', 'AvgH']
    draw_cols = ['B365D', 'PSD', 'MaxD', 'AvgD']
    away_cols = ['B365A', 'PSA', 'MaxA', 'AvgA']
    
    # Extract non-null values
    all_home = [row[c] for c in home_cols if c in row.index and pd.notna(row[c]) and row[c] > 1]
    all_draw = [row[c] for c in draw_cols if c in row.index and pd.notna(row[c]) and row[c] > 1]
    all_away = [row[c] for c in away_cols if c in row.index and pd.notna(row[c]) and row[c] > 1]
    
    # Best (max) odds for each outcome
    best_home = max(all_home) if all_home else row.get('OddsHome', 2.0)
    best_draw = max(all_draw) if all_draw else row.get('OddsDraw', 3.5)
    best_away = max(all_away) if all_away else row.get('OddsAway', 3.0)
    
    # Ensure valid odds
    best_home = max(1.01, float(best_home))
    best_draw = max(1.01, float(best_draw))
    best_away = max(1.01, float(best_away))
    
    return best_home, best_draw, best_away, all_home, all_draw, all_away


def build_dataset(
    horizon_hours: int = 0,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build snapshot dataset from historical data.
    
    Args:
        horizon_hours: Hours before match (0 = closing odds)
        min_date: Optional start date filter
        max_date: Optional end date filter
    
    Returns:
        DataFrame with all features and labels
    """
    # Load data
    df = load_historical_data()
    
    # Date filters
    if min_date:
        df = df[df['Date'] >= min_date]
    if max_date:
        df = df[df['Date'] <= max_date]
    
    logger.info(f"Processing {len(df)} matches for horizon T-{horizon_hours}h")
    
    # Initialize feature builder
    feature_builder = SnapshotFeatureBuilder(df)
    
    # Build features for each match
    rows = []
    skipped = 0
    
    for idx, match in df.iterrows():
        try:
            # Snapshot time = kickoff - horizon
            kickoff_time = match['Date']
            snapshot_time = kickoff_time - timedelta(hours=horizon_hours)
            
            # Get odds
            odds_h, odds_d, odds_a, all_h, all_d, all_a = extract_odds_columns(match)
            
            # Skip if odds are invalid
            if odds_h <= 1.0 or odds_d <= 1.0 or odds_a <= 1.0:
                skipped += 1
                continue
            
            # Build features
            features = make_features(
                home_team=match['HomeTeam'],
                away_team=match['AwayTeam'],
                league=LEAGUE_MAP.get(match['League'], match['League']),
                snapshot_time=snapshot_time,
                odds_home=odds_h,
                odds_draw=odds_d,
                odds_away=odds_a,
                feature_builder=feature_builder,
                all_home_odds=all_h,
                all_draw_odds=all_d,
                all_away_odds=all_a,
            )
            
            # Add metadata
            features['event_id'] = f"{match['HomeTeam']}_{match['AwayTeam']}_{kickoff_time.strftime('%Y%m%d')}"
            features['kickoff_time'] = kickoff_time
            features['snapshot_time'] = snapshot_time
            features['horizon_hours'] = horizon_hours
            
            # Add label
            label_map = {'H': 0, 'D': 1, 'A': 2}
            features['label'] = label_map.get(match['FTR'], -1)
            features['label_str'] = match['FTR']
            
            rows.append(features)
            
        except Exception as e:
            logger.warning(f"Error processing match {idx}: {e}")
            skipped += 1
    
    logger.info(f"Built {len(rows)} samples, skipped {skipped}")
    
    # Convert to DataFrame
    result = pd.DataFrame(rows)
    
    # Ensure column order matches contract
    meta_cols = ['event_id', 'kickoff_time', 'snapshot_time', 'horizon_hours', 'label', 'label_str']
    ordered_cols = meta_cols + FEATURE_ORDER
    
    # Add any missing columns with defaults
    for col in ordered_cols:
        if col not in result.columns:
            if col in FEATURE_COLUMNS:
                dtype = FEATURE_COLUMNS[col]
                if dtype == float:
                    result[col] = 0.0
                elif dtype == int:
                    result[col] = 0
                else:
                    result[col] = 'Unknown'
    
    result = result[ordered_cols]
    
    return result


def save_dataset(df: pd.DataFrame, output_path: Path = OUTPUT_PATH):
    """Save dataset to CSV (avoiding parquet dependency)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use CSV instead of parquet
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved dataset to {csv_path}")
    
    # Also save feature columns for reference
    columns_path = output_path.parent.parent / "models" / "feature_columns.json"
    columns_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(columns_path, 'w') as f:
        json.dump({
            "feature_columns": FEATURE_ORDER,
            "meta_columns": ['event_id', 'kickoff_time', 'snapshot_time', 'horizon_hours', 'label', 'label_str'],
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
        }, f, indent=2)
    logger.info(f"Saved feature contract to {columns_path}")


def print_dataset_stats(df: pd.DataFrame):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("SNAPSHOT DATASET STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal samples: {len(df):,}")
    print(f"Date range: {df['kickoff_time'].min()} to {df['kickoff_time'].max()}")
    
    print("\nLabel distribution:")
    label_counts = df['label_str'].value_counts()
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")
    
    print("\nPer league:")
    for league, count in df['league'].value_counts().items():
        print(f"  {league}: {count:,}")
    
    print("\nFeature summary:")
    for col in FEATURE_ORDER[:10]:  # First 10 features
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            print(f"  {col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build snapshot dataset")
    parser.add_argument("--horizon", type=int, default=0, help="Hours before match")
    parser.add_argument("--min-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--max-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--stats-only", action="store_true", help="Only print stats")
    args = parser.parse_args()
    
    if args.stats_only:
        df = pd.read_parquet(OUTPUT_PATH)
        print_dataset_stats(df)
    else:
        print("\n🔨 BUILDING SNAPSHOT DATASET")
        print("=" * 60)
        
        df = build_dataset(
            horizon_hours=args.horizon,
            min_date=args.min_date,
            max_date=args.max_date,
        )
        
        output = Path(args.output) if args.output else OUTPUT_PATH
        save_dataset(df, output)
        print_dataset_stats(df)
        
        print("\n✅ DATASET BUILD COMPLETE")
