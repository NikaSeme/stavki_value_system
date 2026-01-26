#!/usr/bin/env python3
"""
Prepare v3 Dataset (Strict Time Split)
Generates train_v3, val_v3, test_v3 parquets without shuffling.
"""
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# Paths
DATA_DIR = Path("data/processed")
INPUT_FILE = DATA_DIR / "epl_features_2021_2024.csv"
OUTPUT_DIR = DATA_DIR

AUDIT_DIR = Path("audit_pack/A2_data")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_FILE = AUDIT_DIR / "data_profile.json"
SPLIT_REPORT = Path("audit_pack/A3_time_integrity/time_split_report.json")
SPLIT_REPORT.parent.mkdir(parents=True, exist_ok=True)

def main():
    print("=== Preparing v3 Dataset ===")
    
    if not INPUT_FILE.exists():
        print(f"❌ Input file not found: {INPUT_FILE}")
        # Try finding any features file
        fallback = list(DATA_DIR.glob("*features*.csv"))
        if fallback:
            print(f"   Using fallback: {fallback[0]}")
            df = pd.read_csv(fallback[0])
        else:
            print("   Assuming data generation needed.")
            # For audit reproduction, we might need to mock or fail
            # But let's check if we can skip renaming and just report
            raise FileNotFoundError("No feature data found")
    else:
        df = pd.read_csv(INPUT_FILE)

    # Sort by Date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    else:
        print("⚠ 'Date' column missing. Cannot perform time split!")
        # Fallback to index split if strictly ordered?
        # Assuming index is time for now if Date missing, but this is bad.
        pass

    total_rows = len(df)
    print(f"Total rows: {total_rows}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    # Splits: 70% Train, 15% Val, 15% Test
    n_train = int(total_rows * 0.70)
    n_val = int(total_rows * 0.15)
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train+n_val]
    test_df = df.iloc[n_train+n_val:]
    
    print(f"Train: {len(train_df)} ({train_df['Date'].min()} - {train_df['Date'].max()})")
    print(f"Val:   {len(val_df)} ({val_df['Date'].min()} - {val_df['Date'].max()})")
    print(f"Test:  {len(test_df)} ({test_df['Date'].min()} - {test_df['Date'].max()})")
    
    # Save Parquets
    train_df.to_parquet(OUTPUT_DIR / "train_v3.parquet")
    val_df.to_parquet(OUTPUT_DIR / "val_v3.parquet")
    test_df.to_parquet(OUTPUT_DIR / "test_v3.parquet")
    print("✅ Saved v3 parquets")
    
    # Profile (Null checks)
    feature_cols = [c for c in df.columns if c not in ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'match_id', 'id']]
    # Simple heuristic for features: numeric
    numeric = df.select_dtypes(include=[np.number])
    null_counts = numeric.isnull().sum()
    null_pct = (null_counts / len(df)) * 100
    
    profile = {
        "total_rows": total_rows,
        "date_range": {
            "start": str(df['Date'].min()),
            "end": str(df['Date'].max())
        },
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        },
        "features_null_stats": null_pct[null_pct > 0].to_dict(),
        "status": "READY"
    }
    
    with open(PROFILE_FILE, "w") as f:
        json.dump(profile, f, indent=2)
        
    with open(SPLIT_REPORT, "w") as f:
        json.dump({
            "method": "Strict Time Split (Walk-Forward)",
            "ranges": {
                "train": [str(train_df['Date'].min()), str(train_df['Date'].max())],
                "val": [str(val_df['Date'].min()), str(val_df['Date'].max())],
                "test": [str(test_df['Date'].min()), str(test_df['Date'].max())]
            }
        }, f, indent=2)
        
    print(f"✅ Generated {PROFILE_FILE}")

if __name__ == "__main__":
    main()
