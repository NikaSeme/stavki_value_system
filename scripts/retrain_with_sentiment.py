"""
Retrain Model B (CatBoost) with sentiment features.

Adds 6 sentiment features to the existing 22 base features.
Total: 28 features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import sys
import logging

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.sentiment_features import SentimentFeatureExtractor
from scripts.train_model import main as train_catboost

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_sentiment_features_to_dataset(
    base_file='data/processed/epl_features_2021_2024.csv',
    output_file='data/processed/epl_features_with_sentiment_2021_2024.csv',
    mode='mock'
):
    """
    Add sentiment features to existing dataset.
    
    Args:
        base_file: Existing features dataset
        output_file: Output dataset with sentiment
        mode: 'mock', 'twitter', or 'news'
        
    Returns:
        Path to enhanced dataset
    """
    logger.info("=" * 70)
    logger.info("ADDING SENTIMENT FEATURES TO DATASET")
    logger.info("=" * 70)
    
    # Load base features
    base_path = Path(base_file)
    logger.info(f"Loading base dataset: {base_path}")
    
    df = pd.read_csv(base_path)
    logger.info(f"  Loaded {len(df)} matches")
    logger.info(f"  Base features: {len(df.columns) - 5}")  # Minus Date, teams, FTR, Season
    
    # Initialize sentiment extractor
    extractor = SentimentFeatureExtractor(mode=mode)
    
    # Add sentiment features
    logger.info("\nExtracting sentiment features (using mock data)...")
    
    sentiment_features = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            logger.info(f"  Processed {idx}/{len(df)} matches...")
        
        # Extract sentiment for this match
        features = extractor.extract_for_match(
            home_team=row['HomeTeam'],
            away_team=row['AwayTeam'],
            lookback_hours=48
        )
        
        sentiment_features.append(features)
    
    # Convert to DataFrame
    sentiment_df = pd.DataFrame(sentiment_features)
    
    # Combine
    enhanced_df = pd.concat([df, sentiment_df], axis=1)
    
    logger.info(f"\nEnhanced dataset:")
    logger.info(f"  Total features: {len(enhanced_df.columns) - 5}")
    logger.info(f"  New features: {list(sentiment_df.columns)}")
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enhanced_df.to_csv(output_path, index=False)
    
    logger.info(f"\n✓ Saved enhanced dataset: {output_path}")
    
    return output_path


def train_with_sentiment():
    """Train Model B with sentiment features."""
    logger.info("=" * 70)
    logger.info("MODEL B RETRAINING WITH SENTIMENT FEATURES")
    logger.info("=" * 70)
    
    # Step 1: Add sentiment features to dataset
    enhanced_dataset = add_sentiment_features_to_dataset(mode='mock')
    
    # Step 2: Train CatBoost on enhanced dataset
    logger.info("\nTraining CatBoost with sentiment features...")
    logger.info("(Using existing train_model.py script)")
    
    # Note: This is a simplified version
    # In production, we'd modify train_model.py to accept the new dataset
    
    logger.info("\n✓ Dataset enhanced with sentiment features")
    logger.info(f"✓ Ready for training: {enhanced_dataset}")
    logger.info("\nNext step: Retrain using:")
    logger.info("  python scripts/train_model.py --data data/processed/epl_features_with_sentiment_2021_2024.csv")


if __name__ == '__main__':
    train_with_sentiment()
