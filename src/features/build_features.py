"""
Feature engineering module for building rolling statistics.

CRITICAL: All features must respect temporal ordering to prevent data leakage.
For a match on date D, we use ONLY matches where date < D (strictly before).
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np

from ..logging_setup import get_logger

logger = get_logger(__name__)


def calculate_points(goals_for: int, goals_against: int) -> int:
    """
    Calculate points for a match result.
    
    Args:
        goals_for: Goals scored by the team
        goals_against: Goals conceded by the team
        
    Returns:
        Points: 3 for win, 1 for draw, 0 for loss
    """
    if goals_for > goals_against:
        return 3
    elif goals_for == goals_against:
        return 1
    else:
        return 0


def calculate_team_form(
    matches_df: pd.DataFrame,
    team: str,
    before_date: datetime,
    window: int = 5
) -> Dict[str, float]:
    """
    Calculate rolling statistics for a team using only matches BEFORE a given date.
    
    CRITICAL: This function enforces data leakage prevention by using ONLY
    matches where date < before_date (strictly before, never equal).
    
    Args:
        matches_df: DataFrame with all matches (must have 'date' column as datetime)
        team: Team name to calculate stats for
        before_date: Calculate stats using only matches before this date
        window: Number of recent matches to use (default: 5)
        
    Returns:
        Dictionary with rolling statistics:
        - goals_for_avg: Average goals scored
        - goals_against_avg: Average goals conceded
        - points_avg: Average points per match
        - form_points: Total points in window
        - matches_count: Number of matches used
    """
    # Filter matches involving this team BEFORE the target date
    # CRITICAL: Use strict < comparison to prevent data leakage
    team_matches = matches_df[
        (
            (matches_df['home_team'] == team) | 
            (matches_df['away_team'] == team)
        ) &
        (matches_df['date'] < before_date)  # STRICT < to prevent leakage
    ].copy()
    
    if len(team_matches) == 0:
        # No history available
        return {
            'goals_for_avg': np.nan,
            'goals_against_avg': np.nan,
            'points_avg': np.nan,
            'form_points': np.nan,
            'matches_count': 0
        }
    
    # Sort by date ascending and take last N matches
    team_matches = team_matches.sort_values('date', ascending=True).tail(window)
    
    # Calculate goals for/against from team's perspective
    goals_for = []
    goals_against = []
    points = []
    
    for _, match in team_matches.iterrows():
        if match['home_team'] == team:
            # Team was home
            gf = match['home_goals']
            ga = match['away_goals']
        else:
            # Team was away
            gf = match['away_goals']
            ga = match['home_goals']
        
        goals_for.append(gf)
        goals_against.append(ga)
        points.append(calculate_points(gf, ga))
    
    # Calculate statistics
    return {
        'goals_for_avg': np.mean(goals_for),
        'goals_against_avg': np.mean(goals_against),
        'points_avg': np.mean(points),
        'form_points': np.sum(points),
        'matches_count': len(team_matches)
    }


def build_match_features(
    matches_df: pd.DataFrame,
    window: int = 5
) -> pd.DataFrame:
    """
    Build feature dataset with rolling statistics for all matches.
    
    For each match, calculates features using ONLY previous matches to
    prevent data leakage.
    
    Args:
        matches_df: Normalized matches DataFrame
        window: Rolling window size (default: 5)
        
    Returns:
        DataFrame with original match data plus rolling features
    """
    logger.info(f"Building features with window size {window}")
    
    # Ensure date column is datetime
    if matches_df['date'].dtype != 'datetime64[ns]':
        matches_df['date'] = pd.to_datetime(matches_df['date'])
    
    # Sort by date to ensure temporal ordering
    matches_df = matches_df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"Processing {len(matches_df)} matches chronologically")
    
    # Initialize feature columns
    feature_rows = []
    
    for idx, match in matches_df.iterrows():
        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(matches_df)} matches")
        
        match_date = match['date']
        home_team = match['home_team']
        away_team = match['away_team']
        
        # Calculate home team features (using only matches BEFORE this date)
        home_features = calculate_team_form(
            matches_df,
            home_team,
            before_date=match_date,
            window=window
        )
        
        # Calculate away team features (using only matches BEFORE this date)
        away_features = calculate_team_form(
            matches_df,
            away_team,
            before_date=match_date,
            window=window
        )
        
        # Build feature row
        feature_row = {
            # Original match data
            'date': match['date'],
            'league': match['league'],
            'home_team': match['home_team'],
            'away_team': match['away_team'],
            'home_goals': match['home_goals'],
            'away_goals': match['away_goals'],
            'odds_1': match.get('odds_1', np.nan),
            'odds_x': match.get('odds_x', np.nan),
            'odds_2': match.get('odds_2', np.nan),
            
            # Home team rolling features
            'home_goals_for_avg_5': home_features['goals_for_avg'],
            'home_goals_against_avg_5': home_features['goals_against_avg'],
            'home_points_avg_5': home_features['points_avg'],
            'home_form_points_5': home_features['form_points'],
            'home_matches_count': home_features['matches_count'],
            
            # Away team rolling features
            'away_goals_for_avg_5': away_features['goals_for_avg'],
            'away_goals_against_avg_5': away_features['goals_against_avg'],
            'away_points_avg_5': away_features['points_avg'],
            'away_form_points_5': away_features['form_points'],
            'away_matches_count': away_features['matches_count'],
        }
        
        feature_rows.append(feature_row)
    
    features_df = pd.DataFrame(feature_rows)
    
    logger.info(f"Built features for {len(features_df)} matches")
    logger.info(
        f"Average home matches count: {features_df['home_matches_count'].mean():.1f}"
    )
    logger.info(
        f"Average away matches count: {features_df['away_matches_count'].mean():.1f}"
    )
    
    return features_df


def build_features_dataset(
    input_file: Path,
    output_file: Path,
    window: int = 5
) -> Dict[str, any]:
    """
    Main pipeline: Load matches, build features, save to file.
    
    Args:
        input_file: Path to normalized matches CSV
        output_file: Path to save features CSV
        window: Rolling window size
        
    Returns:
        Dictionary with processing statistics
    """
    logger.info(f"Loading matches from {input_file}")
    
    # Load normalized matches
    try:
        matches_df = pd.read_csv(input_file)
    except Exception as e:
        logger.error(f"Failed to load matches: {e}")
        raise
    
    # Validate required columns
    required_cols = [
        'date', 'league', 'home_team', 'away_team',
        'home_goals', 'away_goals'
    ]
    missing = [col for col in required_cols if col not in matches_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    logger.info(f"Loaded {len(matches_df)} matches")
    
    # Build features
    features_df = build_match_features(matches_df, window=window)
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    features_df.to_csv(output_file, index=False)
    logger.info(f"Saved features to {output_file}")
    
    # Calculate statistics
    stats = {
        'total_matches': len(features_df),
        'matches_with_features': len(features_df[features_df['home_matches_count'] > 0]),
        'matches_without_features': len(features_df[features_df['home_matches_count'] == 0]),
        'avg_home_history': features_df['home_matches_count'].mean(),
        'avg_away_history': features_df['away_matches_count'].mean(),
    }
    
    return stats
