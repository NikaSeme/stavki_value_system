
import pandas as pd
from pathlib import Path

def revert_columns():
    input_path = Path('data/processed/multi_league_combined_2021_2024.csv')
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"Current columns: {df.columns.tolist()}")

    # Map back to PascalCase
    # Note: 'date' -> 'Date', 'odds_1' -> 'OddsHome' (or 'B365H'?)
    # engineer_multi_league_features.py uses: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, League
    # optimize uses: OddsHome, OddsDraw, OddsAway
    
    rename_map = {
        'date': 'Date',
        'home_team': 'HomeTeam',
        'away_team': 'AwayTeam',
        'home_goals': 'FTHG',
        'away_goals': 'FTAG',
        'result': 'FTR',
        'league': 'League',
        'season': 'Season',
        'odds_1': 'OddsHome',
        'odds_x': 'OddsDraw',
        'odds_2': 'OddsAway',
        'odds_home': 'OddsHome',
        'odds_draw': 'OddsDraw', 
        'odds_away': 'OddsAway'
    }
    
    df = df.rename(columns=rename_map)
    
    required = ['Date', 'League', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        print(f"❌ Missing columns after revert: {missing}")
    else:
        print("✅ Column revert successful")
        
    df.to_csv(input_path, index=False)
    print(f"Saved reverted CSV to {input_path}")

if __name__ == "__main__":
    revert_columns()
