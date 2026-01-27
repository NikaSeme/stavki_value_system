
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.build_features import build_features_dataset
from src.features.elo import calculate_elo_for_dataset

def main():
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / 'data/processed/epl_historical_2021_2024.csv'
    temp_features = base_dir / 'data/processed/temp_features_raw.csv'
    final_output = base_dir / 'data/processed/epl_features_2021_2024.csv'
    
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    print("Stage 1: Loading History...")
    df = pd.read_csv(input_file)
    
    # --- UPGRADE: Calculate Elo Ratings ---
    print("Stage 1.5: Injecting Elo Ratings...")
    # Ensure date is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate Elo (Adds HomeEloBefore, AwayEloBefore, EloDiff, etc)
    # Only if columns exist match Elo expectations
    if 'HomeTeam' in df.columns and 'FTR' in df.columns:
        try:
            df = calculate_elo_for_dataset(df)
            print(f"   Elo calculated. Range: {df['HomeEloBefore'].min():.0f}-{df['HomeEloBefore'].max():.0f}")
        except Exception as e:
            print(f"   Warning: Elo calculation failed: {e}")
    # -------------------------------------

    print("Stage 2: Normalizing...")
    rename_map = {
        'Date': 'date',
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'FTHG': 'home_goals',
        'FTAG': 'away_goals',
        'Season': 'league'
    }
    
    df_norm = df.rename(columns=rename_map)
    df_norm['league'] = 'EPL' 
    
    temp_norm = base_dir / 'data/processed/temp_normalized.csv'
    df_norm.to_csv(temp_norm, index=False)
    
    print("Stage 3: Building Rolling Features...")
    stats = build_features_dataset(temp_norm, temp_features)
    print("Stats:", stats)
    
    print("Stage 4: Merging Elo & Translating to Legacy...")
    feat_df = pd.read_csv(temp_features)
    
    # 4.1 Join Elo columns back!
    # We join on 'date', 'home_team', 'away_team'
    # First, let's prepare the Elo lookup
    if 'HomeEloBefore' in df.columns:
        elo_cols = ['Date', 'HomeTeam', 'AwayTeam', 'HomeEloBefore', 'AwayEloBefore', 'EloDiff']
        elo_df = df[elo_cols].copy()
        elo_df = elo_df.rename(columns={'Date': 'date', 'HomeTeam': 'home_team', 'AwayTeam': 'away_team'})
        
        # Convert dates to string for safe merge, or ensure both datetime
        feat_df['date'] = pd.to_datetime(feat_df['date'])
        elo_df['date'] = pd.to_datetime(elo_df['date'])
        
        # Merge
        feat_df = pd.merge(feat_df, elo_df, on=['date', 'home_team', 'away_team'], how='left')
    
    # 4.2 Rename back to Legacy
    reverse_map = {
        'date': 'Date',
        'home_team': 'HomeTeam',
        'away_team': 'AwayTeam',
        'home_goals': 'FTHG',
        'away_goals': 'FTAG',
        'league': 'Season'
    }
    feat_df = feat_df.rename(columns=reverse_map)
    
    # 4.3 Re-calculate FTR
    # Ensure FTHG/FTAG are numeric
    feat_df['FTHG'] = pd.to_numeric(feat_df['FTHG'], errors='coerce')
    feat_df['FTAG'] = pd.to_numeric(feat_df['FTAG'], errors='coerce')
    
    conditions = [
        (feat_df['FTHG'] > feat_df['FTAG']),
        (feat_df['FTHG'] < feat_df['FTAG'])
    ]
    choices = ['H', 'A']
    feat_df['FTR'] = np.select(conditions, choices, default='D')
    
    feat_df.to_csv(final_output, index=False)
    
    print(f"✅ Success! Enhanced Features saved to {final_output}")
    print(f"   Includes Elo: {'HomeEloBefore' in feat_df.columns}")
    print(f"   Includes Date: {'Date' in feat_df.columns}")

    if temp_norm.exists(): temp_norm.unlink()
    if temp_features.exists(): temp_features.unlink()

if __name__ == "__main__":
    main()
