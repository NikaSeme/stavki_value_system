
import pandas as pd
from pathlib import Path

def fix_columns():
    input_path = Path('data/processed/multi_league_combined_2021_2024.csv')
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"Original columns: {df.columns.tolist()}")

    # Map to snake_case expected by build_features
    rename_map = {
        'Date': 'date',
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'FTHG': 'home_goals',
        'FTAG': 'away_goals',
        'FTR': 'result',
        'League': 'league',
        'Season': 'season',
        'OddsHome': 'odds_1',  # build_features expects odds_1
        'OddsDraw': 'odds_x',
        'OddsAway': 'odds_2',
        'odds_home': 'odds_1', # fix for previous run
        'odds_draw': 'odds_x',
        'odds_away': 'odds_2'
    }
    
    # Remove early exit to ensure odds are renamed
    # if 'home_team' in df.columns:
    #     print("Columns already appear to be normalized.")
    #     return

    df = df.rename(columns=rename_map)
    
    # Verify
    required = ['date', 'league', 'home_team', 'away_team', 'home_goals', 'away_goals']
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        print(f"❌ Still missing columns after rename: {missing}")
        # Check what FTHG/FTAG might be named if not standard?
        # download_multi_league_data.py ensures FTHG/FTAG/FTR
    else:
        print("✅ Column mapping successful")
        
    df.to_csv(input_path, index=False)
    print(f"Saved fixed CSV to {input_path}")

if __name__ == "__main__":
    fix_columns()
