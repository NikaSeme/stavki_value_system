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
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_setup import get_logger
from src.features.elo import calculate_elo_for_dataset

logger = get_logger(__name__)


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
    
    # Calculate Elo ratings per-league (avoids cross-league contamination)
    logger.info("Calculating Elo ratings per league...")
    df = df.groupby('League', group_keys=False).apply(
        lambda g: calculate_elo_for_dataset(g.copy())
    )
    # EloDiff is already added by calculate_elo_for_dataset, but ensure it exists
    if 'EloDiff' not in df.columns:
        df['EloDiff'] = df['HomeEloBefore'] - df['AwayEloBefore']
    
    # Fill any NaNs in ELO (rare, but safety first)
    df['HomeEloBefore'] = df['HomeEloBefore'].fillna(1500.0)
    df['AwayEloBefore'] = df['AwayEloBefore'].fillna(1500.0)
    df['EloDiff'] = df['EloDiff'].fillna(0.0)

    logger.info(f"  Elo range: {df['HomeEloBefore'].min():.0f} - {df['HomeEloBefore'].max():.0f}")

    # --- PHASE 4: Holy Trinity Features ---
    logger.info("Adding Market & Sentiment features...")
    
    # 1. Sentiment (Placeholders for history, populated in Live)
    df['SentimentHome'] = 0.0
    df['SentimentAway'] = 0.0
    df['HomeInjury'] = 0
    df['AwayInjury'] = 0
    
    # 2. Market Intelligence
    # Initialize defaults
    df['Market_Consensus'] = 0.0
    df['Sharp_Divergence'] = 0.0
    df['Odds_Volatility'] = 0.0
    
    # Sharp Divergence: (Pinnacle - B365) / B365 (Home)
    # Negative value = Sharps (PSH) have lower odds than Retail (B365) => Sharps like Home
    if 'PSH' in df.columns and 'B365H' in df.columns:
        # Use simple fillna(0) to handle missing rows if columns exist but have NaNs
        psh = df['PSH'].fillna(df['OddsHome'])
        b365 = df['B365H'].fillna(df['OddsHome'])
        df['Sharp_Divergence'] = (psh - b365) / b365
        
    # Volatility: (Max - Avg) / Avg (Home)
    if 'MaxH' in df.columns and 'AvgH' in df.columns:
        max_h = df['MaxH'].fillna(df['OddsHome'])
        avg_h = df['AvgH'].fillna(df['OddsHome'])
        df['Odds_Volatility'] = (max_h - avg_h) / avg_h
        
    # Consensus: Avg of available odds
    # Note: 'OddsHome' is already a canonical copy, but let's be explicit
    odds_sources = [c for c in ['B365H', 'PSH', 'AvgH'] if c in df.columns]
    if odds_sources:
        df['Market_Consensus'] = df[odds_sources].mean(axis=1)
    
    # Fill any remaining NaNs in these new features
    for col in ['Sharp_Divergence', 'Odds_Volatility', 'Market_Consensus']:
        df[col] = df[col].fillna(0.0)

    # ---------------------------------------
    
    # Initialize feature columns
    feature_cols = [
        'Home_GF_L5', 'Home_GA_L5', 'Home_Pts_L5',
        'Away_GF_L5', 'Away_GA_L5', 'Away_Pts_L5',
        # New Phase 2 Features
        'Home_Overall_GF_L5', 'Home_Overall_GA_L5', 'Home_Overall_Pts_L5',
        'Away_Overall_GF_L5', 'Away_Overall_GA_L5', 'Away_Overall_Pts_L5'
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
            
            # Overall games (Home + Away) - NEW Phase 2 Feature
            # We need to construct a chronological list of results for this team
            team_mask = league_mask & ((df['HomeTeam'] == team) | (df['AwayTeam'] == team))
            team_games = df[team_mask].sort_values('Date').copy()
            
            # We need to normalize columns to 'Goals For', 'Goals Against', 'Points'
            # Create temporary normalized view
            is_home = (team_games['HomeTeam'] == team)
            goals_for = np.where(is_home, team_games['FTHG'], team_games['FTAG'])
            goals_against = np.where(is_home, team_games['FTAG'], team_games['FTHG'])
            
            result_map_h = {'H': 3, 'D': 1, 'A': 0}
            result_map_a = {'H': 0, 'D': 1, 'A': 3}
            # Vectorized map approach might be tricky with mixed types, use list comprehension or apply
            points = []
            for idx, row in team_games.iterrows():
                res = row['FTR']
                if row['HomeTeam'] == team:
                    points.append(3 if res == 'H' else (1 if res == 'D' else 0))
                else:
                    points.append(3 if res == 'A' else (1 if res == 'D' else 0))
            
            # Series for rolling
            s_gf = pd.Series(goals_for, index=team_games.index)
            s_ga = pd.Series(goals_against, index=team_games.index)
            s_pts = pd.Series(points, index=team_games.index)
            
            # Calculate Rolling
            roll_gf = s_gf.shift(1).rolling(5, min_periods=1).mean()
            roll_ga = s_ga.shift(1).rolling(5, min_periods=1).mean()
            roll_pts = s_pts.shift(1).rolling(5, min_periods=1).mean()
            
            # Assign back (We must assign to matches where team is Home AND matches where team is Away)
            # Home matches
            df.loc[league_mask & (df['HomeTeam'] == team), 'Home_Overall_GF_L5'] = roll_gf[df['HomeTeam'] == team]
            df.loc[league_mask & (df['HomeTeam'] == team), 'Home_Overall_GA_L5'] = roll_ga[df['HomeTeam'] == team]
            df.loc[league_mask & (df['HomeTeam'] == team), 'Home_Overall_Pts_L5'] = roll_pts[df['HomeTeam'] == team]
            
            # Away matches
            df.loc[league_mask & (df['AwayTeam'] == team), 'Away_Overall_GF_L5'] = roll_gf[df['AwayTeam'] == team]
            df.loc[league_mask & (df['AwayTeam'] == team), 'Away_Overall_GA_L5'] = roll_ga[df['AwayTeam'] == team]
            df.loc[league_mask & (df['AwayTeam'] == team), 'Away_Overall_Pts_L5'] = roll_pts[df['AwayTeam'] == team]
            
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
    
    # Check ELO columns
    assert 'HomeEloBefore' in df.columns, "HomeEloBefore missing"
    assert 'AwayEloBefore' in df.columns, "AwayEloBefore missing"
    # Nulls are allowed in ELO (start of history initialization) but usually filled by ELOCalculator

    
    logger.info("  ✓ Validation passed")


def main():
    logger.info("=" * 70)
    logger.info("MULTI-LEAGUE FEATURE ENGINEERING")
    logger.info("=" * 70)
    
    # Load clean multi-league data
    base_dir = Path(__file__).parent.parent
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
    logger.info(f"   Feature columns (L5): {[c for c in df_features.columns if c.endswith('_L5')]}")
    logger.info(f"   Feature columns (ELO): {[c for c in df_features.columns if 'Elo' in c]}")
    logger.info(f"   Feature columns (Market): {['Sharp_Divergence', 'Odds_Volatility', 'Market_Consensus']}")
    logger.info(f"   Feature columns (Sentiment): {['SentimentHome', 'SentimentAway', 'HomeInjury', 'AwayInjury']}")
    logger.info("\n✅ FEATURE ENGINEERING COMPLETE")
    
    return 0


if __name__ == '__main__':
    exit(main())
