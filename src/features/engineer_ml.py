"""
Feature Engineering for Match Prediction

Creates features for ML model training including:
- Elo ratings
- Recent form (last N games)
- Market features (odds, implied probabilities)
- Home advantage
- Head-to-head history
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_form_features(
    df: pd.DataFrame,
    n_matches: int = 5
) -> pd.DataFrame:
    """
    Calculate recent form features for each team.
    
    Features per team:
    - Points in last N matches
    - Goals scored in last N
    - Goals conceded in last N
    - Win rate in last N
    - Clean sheets in last N
    """
    logger.info(f"Calculating form features (last {n_matches} matches)...")
    
    # Sort by date
    df = df.sort_values('Date').copy()
    
    # Initialize columns
    for prefix in ['Home', 'Away']:
        df[f'{prefix}PointsL{n_matches}'] = 0.0
        df[f'{prefix}GoalsL{n_matches}'] = 0.0
        df[f'{prefix}GoalsAgainstL{n_matches}'] = 0.0
        df[f'{prefix}WinRateL{n_matches}'] = 0.0
        df[f'{prefix}CleanSheetsL{n_matches}'] = 0.0
    
    # Track match history per team
    team_matches = {}
    
    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Initialize team history if needed
        if home_team not in team_matches:
            team_matches[home_team] = []
        if away_team not in team_matches:
            team_matches[away_team] = []
        
        # Calculate form before this match
        # Home team form
        recent_home = team_matches[home_team][-n_matches:]
        if len(recent_home) > 0:
            df.loc[idx, f'HomePointsL{n_matches}'] = sum(m['points'] for m in recent_home)
            df.loc[idx, f'HomeGoalsL{n_matches}'] = sum(m['goals_for'] for m in recent_home)
            df.loc[idx, f'HomeGoalsAgainstL{n_matches}'] = sum(m['goals_against'] for m in recent_home)
            df.loc[idx, f'HomeWinRateL{n_matches}'] = sum(m['won'] for m in recent_home) / len(recent_home)
            df.loc[idx, f'HomeCleanSheetsL{n_matches}'] = sum(m['clean_sheet'] for m in recent_home) / len(recent_home)
        
        # Away team form
        recent_away = team_matches[away_team][-n_matches:]
        if len(recent_away) > 0:
            df.loc[idx, f'AwayPointsL{n_matches}'] = sum(m['points'] for m in recent_away)
            df.loc[idx, f'AwayGoalsL{n_matches}'] = sum(m['goals_for'] for m in recent_away)
            df.loc[idx, f'AwayGoalsAgainstL{n_matches}'] = sum(m['goals_against'] for m in recent_away)
            df.loc[idx, f'AwayWinRateL{n_matches}'] = sum(m['won'] for m in recent_away) / len(recent_away)
            df.loc[idx, f'AwayCleanSheetsL{n_matches}'] = sum(m['clean_sheet'] for m in recent_away) / len(recent_away)
        
        # Update history after this match
        # Home team result
        if row['FTR'] == 'H':
            home_points, home_won = 3, 1
        elif row['FTR'] == 'D':
            home_points, home_won = 1, 0
        else:
            home_points, home_won = 0, 0
        
        team_matches[home_team].append({
            'points': home_points,
            'goals_for': row['FTHG'],
            'goals_against': row['FTAG'],
            'won': home_won,
            'clean_sheet': 1 if row['FTAG'] == 0 else 0,
        })
        
        # Away team result
        if row['FTR'] == 'A':
            away_points, away_won = 3, 1
        elif row['FTR'] == 'D':
            away_points, away_won = 1, 0
        else:
            away_points, away_won = 0, 0
        
        team_matches[away_team].append({
            'points': away_points,
            'goals_for': row['FTAG'],
            'goals_against': row['FTHG'],
            'won': away_won,
            'clean_sheet': 1 if row['FTHG'] == 0 else 0,
        })
    
    logger.info(f"  ✓ Form features calculated")
    return df


def calculate_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate features from betting market odds.
    
    Features:
    - Average odds (home/draw/away)
    - Implied probabilities
    - Odds differentials
    """
    logger.info("Calculating market features...")
    
    # Average odds should already be calculated
    if 'AvgOddsH' not in df.columns:
        logger.warning("  Average odds not found, skipping market features")
        return df
    
    # Implied probabilities from average odds
    df['MarketProbHome'] = 1 / df['AvgOddsH']
    df['MarketProbDraw'] = 1 / df['AvgOddsD']
    df['MarketProbAway'] = 1 / df['AvgOddsA']
    
    # Normalize to remove margin (no-vig probabilities)
    total_prob = df['MarketProbHome'] + df['MarketProbDraw'] + df['MarketProbAway']
    df['MarketProbHomeNoVig'] = df['MarketProbHome'] / total_prob
    df['MarketProbDrawNoVig'] = df['MarketProbDraw'] / total_prob
    df['MarketProbAwayNoVig'] = df['MarketProbAway'] / total_prob
    
    # Odds differentials (favorites vs underdogs)
    df['OddsHomeAwayRatio'] = df['AvgOddsH'] / df['AvgOddsA']
    
    logger.info(f"  ✓ Market features calculated")
    return df


def calculate_h2h_features(
    df: pd.DataFrame,
    n_matches: int = 5
) -> pd.DataFrame:
    """
    Calculate head-to-head history features.
    
    Features:
    - Recent H2H win rate
    - Recent H2H goals for/against
    """
    logger.info(f"Calculating H2H features (last {n_matches} meetings)...")
    
    df = df.sort_values('Date').copy()
    
    # Initialize columns
    df['H2HHomeWins'] = 0.0
    df['H2HDraws'] = 0.0
    df['H2HAwayWins'] = 0.0
    df['H2HHomeGoalsAvg'] = 0.0
    df['H2HAwayGoalsAvg'] = 0.0
    
    # Track H2H history
    h2h_history = {}
    
    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Create key for this matchup
        matchup = tuple(sorted([home_team, away_team]))
        
        if matchup not in h2h_history:
            h2h_history[matchup] = []
        
        # Get recent H2H matches
        recent_h2h = h2h_history[matchup][-n_matches:]
        
        if len(recent_h2h) > 0:
            # Count results from HOME team's perspective
            home_wins = sum(1 for m in recent_h2h if m['winner'] == home_team)
            away_wins = sum(1 for m in recent_h2h if m['winner'] == away_team)
            draws = sum(1 for m in recent_h2h if m['winner'] == 'Draw')
            
            df.loc[idx, 'H2HHomeWins'] = home_wins / len(recent_h2h)
            df.loc[idx, 'H2HDraws'] = draws / len(recent_h2h)
            df.loc[idx, 'H2HAwayWins'] = away_wins / len(recent_h2h)
            
            # Average goals
            home_goals = [m['goals'][home_team] for m in recent_h2h if home_team in m['goals']]
            away_goals = [m['goals'][away_team] for m in recent_h2h if away_team in m['goals']]
            
            if home_goals:
                df.loc[idx, 'H2HHomeGoalsAvg'] = np.mean(home_goals)
            if away_goals:
                df.loc[idx, 'H2HAwayGoalsAvg'] = np.mean(away_goals)
        
        # Update H2H history
        if row['FTR'] == 'H':
            winner = home_team
        elif row['FTR'] == 'A':
            winner = away_team
        else:
            winner = 'Draw'
        
        h2h_history[matchup].append({
            'winner': winner,
            'goals': {home_team: row['FTHG'], away_team: row['FTAG']},
        })
    
    logger.info(f"  ✓ H2H features calculated")
    return df


def create_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create final feature matrix for model training.
    
    Returns:
        DataFrame with features + target
    """
    logger.info("Creating feature matrix...")
    
    # Core features
    feature_cols = [
        # Elo ratings
        'HomeEloBefore', 'AwayEloBefore', 'EloDiff',
        
        # Form features (last 5)
        'HomePointsL5', 'HomeGoalsL5', 'HomeGoalsAgainstL5', 'HomeWinRateL5', 'HomeCleanSheetsL5',
        'AwayPointsL5', 'AwayGoalsL5', 'AwayGoalsAgainstL5', 'AwayWinRateL5', 'AwayCleanSheetsL5',
        
        # Market features
        'MarketProbHomeNoVig', 'MarketProbDrawNoVig', 'MarketProbAwayNoVig',
        'OddsHomeAwayRatio',
        
        # H2H features
        'H2HHomeWins', 'H2HDraws', 'H2HAwayWins',
        'H2HHomeGoalsAvg', 'H2HAwayGoalsAvg',
    ]
    
    # Target
    target_col = 'FTR'
    
    # Select features that exist
    available_features = [col for col in feature_cols if col in df.columns]
    logger.info(f"  Available features: {len(available_features)}/{len(feature_cols)}")
    
    # Create feature matrix
    X = df[available_features].copy()
    y = df[target_col].copy()
    
    # Add metadata columns
    metadata_cols = ['Date', 'HomeTeam', 'AwayTeam', 'Season']
    metadata = df[[col for col in metadata_cols if col in df.columns]].copy()
    
    # Combine
    result = pd.concat([metadata, X, y], axis=1)
    
    # Remove rows with missing features (early matches without form)
    initial_len = len(result)
    result = result.dropna()
    logger.info(f"  Removed {initial_len - len(result)} rows with missing features")
    logger.info(f"  Final dataset: {len(result)} matches")
    
    return result


def main():
    """Build complete feature dataset."""
    # Load data with Elo
    base_dir = Path(__file__).parent.parent.parent
    data_file = base_dir / 'data' / 'processed' / 'epl_with_elo_2021_2024.csv'
    
    logger.info(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate all features
    df = calculate_form_features(df, n_matches=5)
    df = calculate_market_features(df)
    df = calculate_h2h_features(df, n_matches=5)
    
    # Create feature matrix
    df_features = create_feature_matrix(df)
    
    # Save
    output_file = base_dir / 'data' / 'processed' / 'epl_features_2021_2024.csv'
    df_features.to_csv(output_file, index=False)
    logger.info(f"\n✓ Saved feature dataset: {output_file}")
    logger.info(f"  Shape: {df_features.shape}")
    logger.info(f"  Features: {df_features.shape[1] - 5}")  # Subtract metadata + target
    
    # Show feature summary
    feature_cols = [col for col in df_features.columns if col not in ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'FTR']]
    logger.info(f"\nFeature summary:")
    logger.info(f"  Total features: {len(feature_cols)}")
    logger.info(f"  Feature names: {feature_cols[:10]}... (showing first 10)")


if __name__ == '__main__':
    main()
