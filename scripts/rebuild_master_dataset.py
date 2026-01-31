import pandas as pd
import glob
import os
from pathlib import Path

def rebuild_master():
    """
    Merges all 6 league raw CSVs into a unified master dataset with correct odds mapping.
    """
    base_dir = Path("data/raw")
    output_path = Path("data/processed/multi_league_master.csv")
    
    # Mapping raw keys to internal sport_keys
    league_configs = [
        {"dir": "", "pattern": "E0.csv", "key": "soccer_epl"}, 
        {"dir": "championship", "pattern": "*.csv", "key": "soccer_efl_champ"},
        {"dir": "ligue1", "pattern": "*.csv", "key": "soccer_france_ligue_one"},
        {"dir": "seriea", "pattern": "*.csv", "key": "soccer_italy_serie_a"},
        {"dir": "laliga", "pattern": "*.csv", "key": "soccer_spain_la_liga"},
        {"dir": "bundesliga", "pattern": "*.csv", "key": "soccer_germany_bundesliga"},
    ]
    
    all_dfs = []
    
    for conf in league_configs:
        search_path = os.path.join(base_dir, conf['dir'], conf['pattern'])
        files = glob.glob(search_path)
        print(f"Found {len(files)} files for {conf['key']} in {search_path}")
        
        for f in files:
            # We use low_memory=False because these CSVs have mixed types in some columns
            df = pd.read_csv(f, low_memory=False)
            
            # Map Odds (Handle Avg vs B365)
            # Standard columns in FD.co.uk: AvgH, AvgD, AvgA
            if 'AvgH' in df.columns:
                df['AvgOddsH'] = df['AvgH']
                df['AvgOddsD'] = df['AvgD']
                df['AvgOddsA'] = df['AvgA']
            elif 'B365H' in df.columns:
                print(f"  Warning: No AvgH in {f}, falling back to B365")
                df['AvgOddsH'] = df['B365H']
                df['AvgOddsD'] = df['B365D']
                df['AvgOddsA'] = df['B365A']
            
            # Additional fallback columns that Neural/Meta expect
            if 'AvgOddsH' in df.columns:
                df['OddsHome'] = df['AvgOddsH']
                df['OddsDraw'] = df['AvgOddsD']
                df['OddsAway'] = df['AvgOddsA']
            
            # Tag with League Key
            df['League'] = conf['key']
            
            # Select core columns to keep it manageable but functional
            core_cols = [
                'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
                'Season', 'League', 'AvgOddsH', 'AvgOddsD', 'AvgOddsA',
                'OddsHome', 'OddsDraw', 'OddsAway'
            ]
            
            # Add Season if missing (infer from filename or path if possible)
            if 'Season' not in df.columns:
                # E.g. laliga_2021_22.csv -> 2021-22
                basename = os.path.basename(f)
                if '20' in basename:
                    year_part = basename.split('_')[-2] + '-' + basename.split('_')[-1].replace('.csv', '')
                    df['Season'] = year_part
                else:
                    df['Season'] = 'Unknown'
            
            # Keep only available core columns
            available_cols = [c for c in core_cols if c in df.columns]
            all_dfs.append(df[available_cols])
    
    if not all_dfs:
        print("Error: No data found to merge!")
        return
        
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save
    os.makedirs(output_path.parent, exist_ok=True)
    master_df.to_csv(output_path, index=False)
    print(f"Successfully rebuilt master dataset with {len(master_df)} matches: {output_path}")

if __name__ == "__main__":
    rebuild_master()
