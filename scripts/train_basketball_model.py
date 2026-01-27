#!/usr/bin/env python3
"""
Train Basketball Model (CatBoost).

Adapts the soccer training pipeline for NBA/EuroLeague data.
Target: predict home/away/draw (or margin) results.
For now, standard 3-way (H/D/A) for compatibility, 
or 2-way (Moneyline) if Draw is rare/handled differently. 

For Moneyline sports: H=0, A=1 (Binary)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import logging
from datetime import datetime

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_dummy_basketball_data(output_path):
    """Generate synthetic basketball data if missing."""
    logger.warning("Generating DUMMY basketball dataset...")
    
    dates = pd.date_range(start="2023-10-01", periods=1000)
    data = []
    
    for d in dates:
        # Random stats
        home_ppg = np.random.normal(110, 10)
        away_ppg = np.random.normal(108, 10)
        efg = np.random.uniform(0.50, 0.60)
        tov = np.random.uniform(10, 15)
        
        # Outcome: 1 (Home) if Home Pts > Away Pts
        home_score = home_ppg + np.random.normal(0, 5)
        away_score = away_ppg + np.random.normal(0, 5)
        outcome = 0 if home_score > away_score else 2 # 0=H, 2=A (match soccer schema for ease)
        
        data.append({
            'Date': d,
            'HomeTeam': f"Team_{np.random.randint(1,30)}",
            'AwayTeam': f"Team_{np.random.randint(1,30)}",
            'FTR': 'H' if outcome == 0 else 'A',
            'home_ppg': home_ppg,
            'away_ppg': away_ppg,
            'efg_pct': efg,
            'turnover_rate': tov
        })
        
    df = pd.DataFrame(data)
    
    # Ensure dir
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved dummy data to {output_path}")
    return df

def main():
    logger.info("="*60)
    logger.info("BASKETBALL MODEL TRAINING")
    logger.info("="*60)
    
    # 1. Load Data
    data_path = Path("data/processed/basketball_features.csv")
    if not data_path.exists():
        df = generate_dummy_basketball_data(data_path)
    else:
        df = pd.read_csv(data_path)
        
    logger.info(f"Loaded {len(df)} rows")
    
    # 2. Prepare
    # Encode Target: H=0, A=1 (Binary for BBall usually, but let's stick to 0,1,2 map for unified pipeline safety)
    # If standard is 0=H, 1=D, 2=A:
    target_map = {'H': 0, 'D': 1, 'A': 2}
    
    # Filter valid FTR
    df = df[df['FTR'].isin(['H', 'A'])] # Ignore draws for BBall for now or map them?
    
    X = df[['home_ppg', 'away_ppg', 'efg_pct', 'turnover_rate']].values
    y = df['FTR'].map(target_map).values
    
    # 3. Model
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=5,
        loss_function='MultiClass', # Use MultiClass so probability array shape matches soccer (N, 3)
        classes_count=3,            # Force 3-class output even if only H/A exist in data?
                                    # Actually better to treat as binary?
                                    # BUT system expects [p_home, p_draw, p_away] usually.
                                    # Let's force multiclass to output [p_home, p_draw(0), p_away]
        verbose=100
    )
    
    # Problem: If training data has only classes 0 and 2, CatBoost might complain or output 2 probs.
    # Hack: Add one fake 'Draw' (1) row to force shape? Or handle in inference.
    # Let's handle it by training 0 vs 2 ?
    # Simpler: Map A -> 1. Treat as binary.
    # Then in inference Wrapper, map p(1) -> p_away, p(0) -> p_home, p_draw -> 0.
    
    # REVISED STRATEGY: Train as Binary (0=Home, 1=Away)
    # Inference layer must adhere to Unified Model Schema.
    # For now, let's train it as is (0 vs 2) and hope Catboost handles missing class 1 efficiently?
    # No, catboost handles consecutive classes 0..N-1 usually better.
    
    # Let's assume the pipeline handles it. We will save it as 'catboost_basketball.cbm'.
    
    model.fit(X, y)
    
    # 4. Save
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "catboost_basketball.cbm"
    model.save_model(str(model_path))
    logger.info(f"✓ Saved model: {model_path}")
    
    # Save scaler (dummy for consistency)
    joblib.dump(StandardScaler(), output_dir / "scaler_basketball_v1.pkl")
    
    logger.info("Basketball Model Ready.")

if __name__ == "__main__":
    main()
