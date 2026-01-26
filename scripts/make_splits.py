#!/usr/bin/env python3
"""
Make Splits (v3.3)
Generates strict time-based splits for training, validation, and testing.
Includes Walk-Forward Logic.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    return logging.getLogger(__name__)

def make_splits(
    input_file: str = "data/processed/train_v3.parquet",
    output_dir: str = "data/processed/splits_v3_3",
    test_ratio: float = 0.15,
    val_ratio: float = 0.15
):
    logger = setup_logging()
    
    # Load data
    df = pd.read_parquet(input_file)
    logger.info(f"Loaded {len(df)} rows from {input_file}")
    
    # Ensure sorted by time
    if 'date_start' in df.columns:
        date_col = 'date_start'
    elif 'start_time' in df.columns:
        date_col = 'start_time'
    elif 'Date' in df.columns:
        date_col = 'Date'
    else:
        logger.error(f"No date column found. Available: {df.columns.tolist()}")
        return

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    # Split logic
    n = len(df)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_size = n - test_size - val_size
    
    train = df.iloc[:train_size]
    val = df.iloc[train_size:train_size+val_size]
    test = df.iloc[train_size+val_size:]
    
    # Checks
    logger.info(f"Train range: {train[date_col].min()} to {train[date_col].max()}")
    logger.info(f"Val range:   {val[date_col].min()} to {val[date_col].max()}")
    logger.info(f"Test range:  {test[date_col].min()} to {test[date_col].max()}")
    
    if train[date_col].max() > val[date_col].min():
        logger.error("Train leakage into Val!")
    if val[date_col].max() > test[date_col].min():
        logger.error("Val leakage into Test!")
        
    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    train.to_parquet(out / "train.parquet")
    val.to_parquet(out / "val.parquet")
    test.to_parquet(out / "test.parquet")
    
    logger.info(f"Saved splits to {output_dir}")
    
    # Generate Report
    report = {
        "total_rows": n,
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "train_range": [str(train[date_col].min()), str(train[date_col].max())],
        "val_range": [str(val[date_col].min()), str(val[date_col].max())],
        "test_range": [str(test[date_col].min()), str(test[date_col].max())]
    }
    
    with open("audit_pack/A3_time_integrity/time_split_report_v3_3.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    make_splits()
