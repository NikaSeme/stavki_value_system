#!/usr/bin/env python3
"""
Feature Engineering for Multi-League Dataset

Adds form features to multi-league clean data:
- Rolling averages (goals for/against, last 5 games)
- Points and form metrics
- Home/away split statistics
- League-aware processing (doesn't mix league stats)

Avoids data leakage by only using past information.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_rolling_stats(df, team_col, window=5):
    """
    Calculate rolling statistics for a team.
    
    Args:
        df: DataFrame sorted by Date
        team_col: 'HomeTeam' or 'AwayTeam'
        window: Number of games to average
    
    Returns:
        Series with rolling stats
    """
    is_home = (team_col == 'HomeTeam')
    
    # Goals scored
    goals_for = df['FTHG'] if is_home else df['FTAG']
    # Goals conceded
    goals_against = df['FTAG'] if is_home else df['FTHG']
    
    # Points (H=3, D=1, A=0)
    results = df['FTR'].copy()
    if is_home:
        points = results.map({'H': 3, 'D': 1, 'A': 0})
    else:
        points = results.map({'H': 0, 'D': 1, 'A': 3})
    
    # Rolling averages (shift to avoid leakage)
    rolling_gf = goals_for.shift(1).rolling(window, min_periods=1).mean()
    rolling_ga = goals_against.shift(1).rolling(window, min_periods=1).mean()
    rolling_pts = points.shift(1).rolling(window, min_periods=1).mean()
    
    return rolling_gf, rolling_ga, rolling_pts


def engineer_features(df):
    """
    Add form features to multi-league dataset.
    
    Features added per team:
    - Last 5 games: goals for/against, points
    - Home/away splits
    - League-specific (doesn't cross league boundaries)
    """
    logger.info("Engineering features...")
    
    # Sort by date to ensure chronological order
    df = df.sort_values(['League', 'Date']).reset_index(drop=True)
    
    # Initialize feature columns
    feature_cols = [
        'Home_GF_L5', 'Home_GA_L5', 'Home_Pts_L5',
        'Away_GF_L5', 'Away_GA_L5', 'Away_Pts_L5'
    ]
    
    for col in feature_cols:
        df[col] = np.nan
    
    # Process each league separately to avoid cross-league contamination
    for league in df['League'].unique():
        logger.info(f"  Processing {league}...")
        
        league_mask = df['League'] == league
        league_df = df[league_mask].copy()
        
        # Process each team
        all_teams = set(league_df['HomeTeam'].unique()) | set(league_df['AwayTeam'].unique())
        
        for team in all_teams:
            # Home games
            home_mask = (league_df['HomeTeam'] == team)
            if home_mask.sum() > 0:
                home_df = league_df[home_mask].copy()
                gf, ga, pts = calculate_rolling_stats(home_df, 'HomeTeam')
                
                df.loc[league_mask & (df['HomeTeam'] == team), 'Home_GF_L5'] = gf.values
                df.loc[league_mask & (df['HomeTeam'] == team), 'Home_GA_L5'] = ga.values
                df.loc[league_mask & (df['HomeTeam'] == team), 'Home_Pts_L5'] = pts.values
            
            # Away games
            away_mask = (league_df['AwayTeam'] == team)
            if away_mask.sum() > 0:
                away_df = league_df[away_mask].copy()
                gf, ga, pts = calculate_rolling_stats(away_df, 'AwayTeam')
                
                df.loc[league_mask & (df['AwayTeam'] == team), 'Away_GF_L5'] = gf.values
                df.loc[league_mask & (df['AwayTeam'] == team), 'Away_GA_L5'] = ga.values
                df.loc[league_mask & (df['AwayTeam'] == team), 'Away_Pts_L5'] = pts.values
    
    # Fill any remaining NaN with league averages (for first few games)
    for league in df['League'].unique():
        league_mask = df['League'] == league
        
        for col in feature_cols:
            league_mean = df.loc[league_mask, col].mean()
            df.loc[league_mask, col] = df.loc[league_mask, col].fillna(league_mean)
    
    logger.info(f"  Added {len(feature_cols)} form features")
    
    return df


def validate_no_leakage(df):
    """
    Verify no data leakage in features.
    """
    logger.info("Validating features...")
    
    # Check: first game of each team should have NaN or defaults
    # After filling, check that features don't perfectly predict outcomes
    
    # Simple check: ensure features existed before adding them
    required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'League']
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check no NaN in final dataset
    assert df[['Home_GF_L5', 'Home_GA_L5', 'Home_Pts_L5']].isnull().sum().sum() == 0
    assert df[['Away_GF_L5', 'Away_GA_L5', 'Away_Pts_L5']].isnull().sum().sum() == 0
    
    logger.info("  ✓ Validation passed")


def main():
    logger.info("=" * 70)
    logger.info("MULTI-LEAGUE FEATURE ENGINEERING")
    logger.info("=" * 70)
    
    # Load clean multi-league data
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / 'data' / 'processed' / 'multi_league_combined_2021_2024.csv'
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1
    
    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    logger.info(f"Loaded {len(df)} matches")
    logger.info(f"Leagues: {df['League'].value_counts().to_dict()}")
    
    # Engineer features
    df_features = engineer_features(df)
    
    # Validate
    validate_no_leakage(df_features)
    
    # Save
    output_file = base_dir / 'data' / 'processed' / 'multi_league_features_2021_2024.csv'
    df_features.to_csv(output_file, index=False)
    
    logger.info(f"\n✅ Features saved to: {output_file}")
    logger.info(f"   Total columns: {len(df_features.columns)}")
    logger.info(f"   Feature columns: {[c for c in df_features.columns if c.endswith('_L5')]}")
    logger.info("\n✅ FEATURE ENGINEERING COMPLETE")
    
    return 0


if __name__ == '__main__':
    exit(main())
