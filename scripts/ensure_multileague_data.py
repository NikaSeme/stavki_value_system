#!/usr/bin/env python3
"""
Ensure Multi-League Data (M06)
Fetch odds for all active leagues to ensure artifacts exist.
"""
import sys
import yaml
import subprocess
from pathlib import Path

def main():
    print("Checking Multi-League Data Artifacts...")
    
    # Load config
    try:
        with open("config/leagues.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    leagues = []
    for sport_type in ['soccer', 'basketball']:
        for league in config.get(sport_type, []):
            if league.get('active'):
                leagues.append(league['key'])

    print(f"Active Targeted Leagues: {len(leagues)}")
    for l in leagues:
        print(f" - {l}")
        
    print("\nTriggering Odds Pipeline for all leagues...")
    # We can run the pipeline with no args, it defaults to all active leagues in config
    cmd = ["python3", "scripts/run_odds_pipeline.py"]
    
    try:
        # Run with tracking enabled to satisfy M04 too
        subprocess.run(cmd + ["--track-lines"], check=True)
        print("\n✓ Odds fetching complete.")
    except subprocess.CalledProcessError:
        print("\n✗ Odds fetching failed.")
        sys.exit(1)

    print("\nEnsuring Model Artifacts...")
    # Check if we have models. 
    # Current system uses 'catboost_soccer_v3_3.cbm' for all soccer?
    # And we need basketball support.
    
    models_dir = Path("models")
    basketball_model = models_dir / "catboost_basketball.cbm"
    
    if not basketball_model.exists():
        print("⚠ Basketball model missing (M07/M06).")
        print("For M06 compliance, we should ensure at least placeholder or trained model exists.")
        # We can't train without data.
        # But we just ran odds pipeline. 
        # To truly train, we need HISTORICAL data, not just live odds.
        # This script ensures LIVE data flows. 
        pass
    else:
        print("✓ Basketball model found.")

if __name__ == "__main__":
    main()
