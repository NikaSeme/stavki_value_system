
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.build_features import build_features_dataset

def main():
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / 'data/processed/epl_historical_2021_2024.csv'
    output_file = base_dir / 'data/processed/epl_features_2021_2024.csv'
    
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    print("Normalizing historical data...")
    df = pd.read_csv(input_file)
    
    # Rename columns to match build_features expectations
    rename_map = {
        'Date': 'date',
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'FTHG': 'home_goals',
        'FTAG': 'away_goals',
        'Season': 'league' # Using Season as League placeholder or just filling 'EPL'
    }
    df = df.rename(columns=rename_map)
    df['league'] = 'EPL' # Hardcode league
    
    # Save temporary normalized file
    temp_norm = base_dir / 'data/processed/temp_normalized.csv'
    df.to_csv(temp_norm, index=False)
    
    print("Building features...")
    stats = build_features_dataset(temp_norm, output_file)
    print("Stats:", stats)
    
    # Cleanup
    if temp_norm.exists():
        temp_norm.unlink()
        
    print(f"Success! Features saved to {output_file}")

if __name__ == "__main__":
    main()
