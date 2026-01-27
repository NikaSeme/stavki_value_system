#!/usr/bin/env python3
"""
Train Line Movement Sequence Model.

Reads from data/odds/odds_timeseries.db, creates sequences,
and trains a model to predict sharp drops.
"""

import sys
import argparse
import sqlite3
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sequence_model import LineSequenceModel
from src.data.odds_tracker import OddsTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_sequences_from_db(db_path, n_lags=5):
    """
    Extract training sequences from DB.
    
    Returns:
        sequences: List of arrays (odds history)
        labels: List of ints (1 if sharp drop followed, 0 otherwise)
    """
    conn = sqlite3.connect(db_path)
    # Get all match/outcome pairs
    query = "SELECT DISTINCT match_id, outcome FROM odds_history"
    pairs = pd.read_sql(query, conn)
    conn.close()
    
    sequences = []
    labels = []
    
    tracker = OddsTracker(db_path)
    
    logger.info(f"Processing {len(pairs)} lines for sequences...")
    
    for _, row in pairs.iterrows():
        mid = row['match_id']
        outcome = row['outcome']
        
        # Get full history
        history = tracker.get_line_movement(mid, outcome)
        if len(history) < n_lags + 1:
            continue
            
        # Sort by time
        history.sort(key=lambda x: x[0])
        odds_values = [h[1] for h in history]
        
        # Sliding window
        # Input: t-N...t
        # Target: t+1 (Drop > threshold?)
        
        for i in range(len(odds_values) - n_lags):
            seq = odds_values[i : i+n_lags]
            next_val = odds_values[i+n_lags]
            current_val = seq[-1]
            
            # Label: Did odds drop significantly?
            # Drop > 5% means (next - current) / current < -0.05
            pct_change = (next_val - current_val) / current_val
            target = 1 if pct_change < -0.05 else 0
            
            sequences.append(seq)
            labels.append(target)
            
    return sequences, labels

def main():
    parser = argparse.ArgumentParser(description="Train Sequence Model")
    parser.add_argument('--db', default='data/odds/odds_timeseries.db', help='Path to odds DB')
    parser.add_argument('--output', default='models/sequence_model_v1.pkl', help='Output model path')
    parser.add_argument('--dry-run', action='store_true', help='Use synthetic data if DB empty')
    args = parser.parse_args()
    
    db_path = Path(args.db)
    
    if not db_path.exists() and not args.dry_run:
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)
        
    logger.info("Fetching data...")
    try:
        if args.dry_run and not db_path.exists():
            logger.warning("Dry Run: Generating synthetic data...")
            # Synthetic: 100 sequences of length 5
            sequences = [np.random.normal(2.0, 0.1, 5).tolist() for _ in range(100)]
            labels = np.random.randint(0, 2, 100).tolist()
        else:
            sequences, labels = fetch_sequences_from_db(db_path)
    except Exception as e:
        if args.dry_run:
             logger.warning(f"DB read failed ({e}), falling back to synthetic.")
             sequences = [np.random.normal(2.0, 0.1, 5).tolist() for _ in range(100)]
             labels = np.random.randint(0, 2, 100).tolist()
        else:
            raise e

    if not sequences:
        if args.dry_run:
             logger.warning("No data found, using synthetic.")
             sequences = [np.random.normal(2.0, 0.1, 5).tolist() for _ in range(100)]
             labels = np.random.randint(0, 2, 100).tolist()
        else:
            logger.error("No sequences found in DB. Run tracker to collect data.")
            sys.exit(1)
            
    logger.info(f"Training on {len(sequences)} sequences...")
    model = LineSequenceModel(n_lags=5)
    model.fit(sequences, labels)
    
    logger.info("Saving model...")
    model.save(args.output)
    logger.info(f"✓ Model saved to {args.output}")

if __name__ == "__main__":
    main()
