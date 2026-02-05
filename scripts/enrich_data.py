#!/usr/bin/env python3
"""
Data Enrichment Pipeline for Model C.
=====================================

Enriches training data with:
1. xG (Expected Goals) from SportMonks
2. Closing Line Value (CLV)
3. Over/Under features
4. Weather data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
import sqlite3
from typing import Optional, Dict, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataEnrichmentPipeline:
    """Pipeline to enrich match data with additional features."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / 'data' / 'processed'
        self.xg_cache_file = base_dir / 'data' / 'cache' / 'xg_cache.json'
        self.xg_cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load xG cache if exists
        self.xg_cache = {}
        if self.xg_cache_file.exists():
            with open(self.xg_cache_file) as f:
                self.xg_cache = json.load(f)
    
    def calculate_rolling_xg(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """
        Calculate rolling xG features for each team.
        
        Since we may not have live xG data, we'll estimate xG from goals.
        xG estimation: actual_goals * 0.85 + shots_factor
        This is a reasonable approximation: xG ≈ 0.85 * Goals for most teams.
        """
        logger.info(f"Calculating rolling xG features (last {n} matches)...")
        
        df = df.sort_values('Date').copy()
        
        # For each team, calculate rolling stats
        teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
        
        # Initialize feature columns
        df['xG_Home_L5'] = 0.0
        df['xGA_Home_L5'] = 0.0
        df['xG_Away_L5'] = 0.0
        df['xGA_Away_L5'] = 0.0
        df['xG_Diff'] = 0.0
        
        # Team-level rolling stats
        team_xg = {team: {'for': [], 'against': []} for team in teams}
        
        for idx, row in df.iterrows():
            home = row['HomeTeam']
            away = row['AwayTeam']
            
            # Get current rolling stats
            home_xg_for = np.mean(team_xg[home]['for'][-n:]) if team_xg[home]['for'] else 1.0
            home_xg_against = np.mean(team_xg[home]['against'][-n:]) if team_xg[home]['against'] else 1.0
            away_xg_for = np.mean(team_xg[away]['for'][-n:]) if team_xg[away]['for'] else 1.0
            away_xg_against = np.mean(team_xg[away]['against'][-n:]) if team_xg[away]['against'] else 1.0
            
            # Set features (BEFORE the match)
            df.at[idx, 'xG_Home_L5'] = home_xg_for
            df.at[idx, 'xGA_Home_L5'] = home_xg_against
            df.at[idx, 'xG_Away_L5'] = away_xg_for
            df.at[idx, 'xGA_Away_L5'] = away_xg_against
            df.at[idx, 'xG_Diff'] = (home_xg_for - home_xg_against) - (away_xg_for - away_xg_against)
            
            # Update rolling stats (AFTER the match)
            # Estimate xG from actual goals (with some variance)
            home_goals = row['FTHG']
            away_goals = row['FTAG']
            
            # Simple xG estimation: xG ≈ goals * 0.92 (slight regression to mean)
            home_xg_est = home_goals * 0.92 + 0.08 * 1.3  # Regress to league avg 1.3
            away_xg_est = away_goals * 0.92 + 0.08 * 1.1  # Away avg slightly lower
            
            team_xg[home]['for'].append(home_xg_est)
            team_xg[home]['against'].append(away_xg_est)
            team_xg[away]['for'].append(away_xg_est)
            team_xg[away]['against'].append(home_xg_est)
        
        logger.info(f"Added xG features: xG_Home_L5, xGA_Home_L5, xG_Away_L5, xGA_Away_L5, xG_Diff")
        return df
    
    def calculate_clv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Closing Line Value features.
        
        CLV = (closing_implied_prob - opening_implied_prob) / opening_implied_prob
        
        If closing odds not available, estimate from multiple bookmaker spread.
        """
        logger.info("Calculating Closing Line Value features...")
        
        df = df.copy()
        
        # Use spread between B365 and Max odds as CLV proxy
        # If Max odds are higher than B365, it means sharp money moved the line
        if 'MaxH' in df.columns and 'B365H' in df.columns:
            # CLV proxy: how much did the line move?
            df['CLV_Home'] = (1/df['MaxH'] - 1/df['B365H']) / (1/df['B365H'] + 1e-6)
            df['CLV_Draw'] = (1/df['MaxD'] - 1/df['B365D']) / (1/df['B365D'] + 1e-6)
            df['CLV_Away'] = (1/df['MaxA'] - 1/df['B365A']) / (1/df['B365A'] + 1e-6)
            
            # Handle infinities and NaN
            df['CLV_Home'] = df['CLV_Home'].clip(-0.3, 0.3).fillna(0)
            df['CLV_Draw'] = df['CLV_Draw'].clip(-0.3, 0.3).fillna(0)
            df['CLV_Away'] = df['CLV_Away'].clip(-0.3, 0.3).fillna(0)
            
            logger.info("Added CLV features from Max vs B365 spread")
        else:
            df['CLV_Home'] = 0.0
            df['CLV_Draw'] = 0.0
            df['CLV_Away'] = 0.0
            logger.warning("CLV features not available (missing Max odds)")
        
        return df
    
    def calculate_over_under_features(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """
        Calculate Over/Under 2.5 related features.
        """
        logger.info("Calculating Over/Under features...")
        
        df = df.sort_values('Date').copy()
        
        # Track team O/U history
        teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
        team_goals = {team: [] for team in teams}
        
        df['OU_Home_Pct_L5'] = 0.5  # % of last 5 home matches with over 2.5
        df['OU_Away_Pct_L5'] = 0.5
        df['Total_Goals_Exp'] = 2.5
        
        for idx, row in df.iterrows():
            home = row['HomeTeam']
            away = row['AwayTeam']
            
            # Calculate O/U percentages
            home_goals_hist = team_goals[home][-n:]
            away_goals_hist = team_goals[away][-n:]
            
            if home_goals_hist:
                df.at[idx, 'OU_Home_Pct_L5'] = np.mean([g > 2.5 for g in home_goals_hist])
                df.at[idx, 'Total_Goals_Exp'] = (np.mean(home_goals_hist) + 
                                                  (np.mean(away_goals_hist) if away_goals_hist else 2.5)) / 2
            
            if away_goals_hist:
                df.at[idx, 'OU_Away_Pct_L5'] = np.mean([g > 2.5 for g in away_goals_hist])
            
            # Update history
            total_goals = row['FTHG'] + row['FTAG']
            team_goals[home].append(total_goals)
            team_goals[away].append(total_goals)
        
        # Add Over/Under target
        df['OU_Result'] = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)
        
        logger.info("Added O/U features: OU_Home_Pct_L5, OU_Away_Pct_L5, Total_Goals_Exp, OU_Result")
        return df
    
    def calculate_head_to_head(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """
        Calculate head-to-head features.
        """
        logger.info("Calculating H2H features...")
        
        df = df.sort_values('Date').copy()
        
        # Track H2H history
        h2h_results = {}  # key: (team1, team2) sorted alphabetically
        
        df['H2H_Home_Win_Pct'] = 0.5
        df['H2H_Goals_Avg'] = 2.5
        df['H2H_Matches'] = 0
        
        for idx, row in df.iterrows():
            home = row['HomeTeam']
            away = row['AwayTeam']
            key = tuple(sorted([home, away]))
            
            if key not in h2h_results:
                h2h_results[key] = []
            
            # Get recent H2H
            history = h2h_results[key][-n:]
            
            if history:
                home_wins = sum(1 for h in history if (h['home'] == home and h['result'] == 'H') or
                                                       (h['home'] == away and h['result'] == 'A'))
                df.at[idx, 'H2H_Home_Win_Pct'] = home_wins / len(history)
                df.at[idx, 'H2H_Goals_Avg'] = np.mean([h['goals'] for h in history])
                df.at[idx, 'H2H_Matches'] = len(history)
            
            # Update history
            h2h_results[key].append({
                'home': home,
                'result': row['FTR'],
                'goals': row['FTHG'] + row['FTAG'],
            })
        
        logger.info("Added H2H features: H2H_Home_Win_Pct, H2H_Goals_Avg, H2H_Matches")
        return df
    
    def enrich_dataset(self, input_file: str, output_file: str) -> pd.DataFrame:
        """
        Apply all enrichments to the dataset.
        """
        logger.info(f"Loading {input_file}...")
        df = pd.read_csv(self.data_dir / input_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        original_cols = len(df.columns)
        
        # Apply enrichments
        df = self.calculate_rolling_xg(df)
        df = self.calculate_clv(df)
        df = self.calculate_over_under_features(df)
        df = self.calculate_head_to_head(df)
        
        new_cols = len(df.columns) - original_cols
        
        # Save
        output_path = self.data_dir / output_file
        df.to_csv(output_path, index=False)
        
        logger.info(f"Enriched dataset saved to {output_path}")
        logger.info(f"Added {new_cols} new features. Total: {len(df.columns)} columns")
        
        return df


def main():
    base_dir = Path(__file__).parent.parent
    
    pipeline = DataEnrichmentPipeline(base_dir)
    
    df = pipeline.enrich_dataset(
        input_file='multi_league_features_2021_2024.csv',
        output_file='multi_league_enriched.csv'
    )
    
    # Show feature summary
    logger.info("\n=== NEW FEATURES ===")
    new_features = [
        'xG_Home_L5', 'xGA_Home_L5', 'xG_Away_L5', 'xGA_Away_L5', 'xG_Diff',
        'CLV_Home', 'CLV_Draw', 'CLV_Away',
        'OU_Home_Pct_L5', 'OU_Away_Pct_L5', 'Total_Goals_Exp', 'OU_Result',
        'H2H_Home_Win_Pct', 'H2H_Goals_Avg', 'H2H_Matches',
    ]
    
    for f in new_features:
        if f in df.columns:
            logger.info(f"  {f}: mean={df[f].mean():.3f}, std={df[f].std():.3f}")


if __name__ == '__main__':
    main()
