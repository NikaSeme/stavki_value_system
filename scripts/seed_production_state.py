#!/usr/bin/env python3
"""
Seed Production State for Multi-League Support.

This script replays the entire historical dataset (2021-2024) through the 
LiveFeatureExtractor. This populates the internal state (Elo ratings, Team Form)
so that the production bot has "memory" of past performance for Bundesliga and La Liga.

Usage:
    python scripts/seed_production_state.py
"""

import sys
import pandas as pd
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.live_extractor import LiveFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 70)
    logger.info("SEEDING PRODUCTION STATE (Full Replay)")
    logger.info("=" * 70)
    
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / 'data' / 'processed' / 'multi_league_clean_2021_2024.csv'
    state_file = base_dir / 'data' / 'live_extractor_state.pkl'
    
    if not data_file.exists():
        logger.error(f"❌ Data file missing: {data_file}")
        logger.error("Run scripts/download_multi_league_data.py first.")
        return 1
        
    # Load historical data
    logger.info(f"Loading history from {data_file}...")
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} matches.")
    
    # Sort chronologically (CRITICAL)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    logger.info(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Initialize Extractor with target state file
    extractor = LiveFeatureExtractor(state_file=state_file)
    
    # Reset internal state to ensure clean build
    extractor.elo.ratings = {}
    extractor.team_form = {}
    extractor.elo.history = []
    
    logger.info("Replaying matches...")
    count = 0
    for idx, row in df.iterrows():
        # Update state (Elo + Form)
        # We process matches sequentially as if they just finished
        extractor.update_after_match(
            home_team=row['HomeTeam'],
            away_team=row['AwayTeam'],
            result=row['FTR'],
            goals_home=int(row['FTHG']),
            goals_away=int(row['FTAG'])
        )
        count += 1
        
        if count % 500 == 0:
            print(f"  Processed {count}/{len(df)} matches...", end='\r')
            
    print(f"  Processed {count}/{len(df)} matches... Done.")
    
    # Save final state
    extractor.save_state()
    logger.info(f"\n✅ State saved to {state_file}")
    
    # Validation check
    logger.info("\nValidation:")
    logger.info(f"  Teams with Elo: {len(extractor.elo.ratings)}")
    logger.info(f"  Teams with Form: {len(extractor.team_form)}")
    
    # Check a few top teams
    examples = ['Man City', 'Real Madrid', 'Bayern Munich']
    for team in examples:
        if team in extractor.elo.ratings:
            pts = extractor.elo.get_rating(team)
            form = extractor._get_form_features(team)
            logger.info(f"  {team}: Elo={pts:.0f}, FormPts(L5)={form['points']}")
        else:
            logger.warning(f"  {team}: Not found (Name mismatch?)")

    return 0

if __name__ == '__main__':
    exit(main())
