import pandas as pd
import json
import os

# Configuration
DATA_FILE = 'data/processed/epl_features_2021_2024.csv'
OUTPUT_DIR = 'outputs/audit_v2/A2_data'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading data from {DATA_FILE}...")
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} rows.")

    # 1. Save Sample
    sample_file = os.path.join(OUTPUT_DIR, 'data_sample.csv')
    df.head(50).to_csv(sample_file, index=False)
    print(f"Saved sample to {sample_file}")

    # 2. Generate Profile
    profile = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "date_range": {
            "start": str(df['date'].min()) if 'date' in df.columns else "N/A",
            "end": str(df['date'].max()) if 'date' in df.columns else "N/A"
        },
        "missing_values": df.isnull().sum().to_dict(),
        "unique_values": {
            "season": df['season'].unique().tolist() if 'season' in df.columns else [],
            "home_team": len(df['home_team'].unique()) if 'home_team' in df.columns else 0,
            "away_team": len(df['away_team'].unique()) if 'away_team' in df.columns else 0
        }
    }

    profile_file = os.path.join(OUTPUT_DIR, 'data_profile.json')
    with open(profile_file, 'w') as f:
        json.dump(profile, f, indent=2)
    print(f"Saved profile to {profile_file}")

except FileNotFoundError:
    print(f"ERROR: File {DATA_FILE} not found!")
    # Create empty artifacts so audit doesn't crash on missing file, but flags it
    with open(os.path.join(OUTPUT_DIR, 'data_profile.json'), 'w') as f:
        json.dump({"error": "file_not_found"}, f)
except Exception as e:
    print(f"ERROR: {e}")
