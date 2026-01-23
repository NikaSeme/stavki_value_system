#!/usr/bin/env python3
"""
Download historical football data from Football-Data.co.uk

Downloads match results and odds data for model training.
"""

import os
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Football-Data.co.uk URLs for English Premier League
FOOTBALL_DATA_URLS = {
    '2021-22': 'https://www.football-data.co.uk/mmz4281/2122/E0.csv',
    '2022-23': 'https://www.football-data.co.uk/mmz4281/2223/E0.csv',
    '2023-24': 'https://www.football-data.co.uk/mmz4281/2324/E0.csv',
}


def download_season_data(season: str, url: str, output_dir: Path) -> pd.DataFrame:
    """
    Download CSV data for a single season.
    
    Args:
        season: Season identifier (e.g., '2021-22')
        url: URL to download from
        output_dir: Directory to save raw CSV
        
    Returns:
        DataFrame with season data
    """
    logger.info(f"Downloading {season} data from {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save raw CSV
        output_file = output_dir / f'epl_{season.replace("-", "_")}.csv'
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"  ✓ Saved to {output_file}")
        
        # Load into DataFrame
        df = pd.read_csv(output_file, encoding='latin1')
        df['Season'] = season
        
        logger.info(f"  ✓ Loaded {len(df)} matches")
        
        return df
        
    except Exception as e:
        logger.error(f"  ✗ Failed to download {season}: {e}")
        return None


def clean_football_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize Football-Data.co.uk format.
    
    Key columns:
    - Date: Match date
    - HomeTeam, AwayTeam: Team names
    - FTHG, FTAG: Full-time goals (home/away)
    - FTR: Full-time result (H/D/A)
    - B365H, B365D, B365A: Bet365 odds (home/draw/away)
    - ... (other bookmakers: BW, IW, PS, WH, VC, etc.)
    """
    # Select core columns
    core_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'Season']
    
    # Odds columns (multiple bookmakers)
    odds_cols = []
    for bookmaker in ['B365', 'BW', 'IW', 'PS', 'WH', 'VC']:
        for outcome in ['H', 'D', 'A']:
            col = f'{bookmaker}{outcome}'
            if col in df.columns:
                odds_cols.append(col)
    
    # Half-time scores (optional)
    optional_cols = ['HTHG', 'HTAG', 'HTR']
    optional_cols = [c for c in optional_cols if c in df.columns]
    
    # Select available columns
    selected_cols = core_cols + odds_cols + optional_cols
    selected_cols = [c for c in selected_cols if c in df.columns]
    
    df_clean = df[selected_cols].copy()
    
    # Convert date
    df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%d/%m/%Y', errors='coerce')
    
    # Remove rows with missing core data
    df_clean = df_clean.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTR'])
    
    # Sort by date
    df_clean = df_clean.sort_values('Date').reset_index(drop=True)
    
    logger.info(f"  ✓ Cleaned: {len(df_clean)} matches, {len(selected_cols)} columns")
    
    return df_clean


def calculate_average_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average odds across bookmakers.
    """
    # Find all bookmaker columns
    bookmakers = []
    for col in df.columns:
        if col.endswith('H') and len(col) <= 6 and col not in ['FTHG', 'HTHG']:
            bookmaker = col[:-1]
            if f'{bookmaker}D' in df.columns and f'{bookmaker}A' in df.columns:
                bookmakers.append(bookmaker)
    
    logger.info(f"  Found {len(bookmakers)} bookmakers: {bookmakers}")
    
    # Calculate average odds
    for outcome in ['H', 'D', 'A']:
        outcome_cols = [f'{bk}{outcome}' for bk in bookmakers]
        outcome_cols = [c for c in outcome_cols if c in df.columns]
        
        if outcome_cols:
            df[f'AvgOdds{outcome}'] = df[outcome_cols].mean(axis=1)
    
    return df


def main():
    """Download and process historical football data."""
    # Setup directories
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / 'data' / 'raw' / 'football_data'
    processed_dir = base_dir / 'data' / 'processed'
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("FOOTBALL DATA DOWNLOADER")
    logger.info("=" * 70)
    
    # Download all seasons
    all_data = []
    for season, url in FOOTBALL_DATA_URLS.items():
        df = download_season_data(season, url, raw_dir)
        if df is not None:
            all_data.append(df)
    
    if not all_data:
        logger.error("No data downloaded!")
        return
    
    # Combine all seasons
    logger.info(f"\nCombining {len(all_data)} seasons...")
    df_combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"  ✓ Total matches: {len(df_combined)}")
    
    # Clean data
    logger.info("\nCleaning data...")
    df_clean = clean_football_data(df_combined)
    
    # Calculate average odds
    logger.info("\nCalculating average odds...")
    df_clean = calculate_average_odds(df_clean)
    
    # Save processed data
    output_file = processed_dir / 'epl_historical_2021_2024.csv'
    df_clean.to_csv(output_file, index=False)
    logger.info(f"\n✓ Saved processed data: {output_file}")
    
    # Summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total matches:     {len(df_clean)}")
    logger.info(f"Date range:        {df_clean['Date'].min()} to {df_clean['Date'].max()}")
    logger.info(f"Seasons:           {sorted(df_clean['Season'].unique())}")
    logger.info(f"Teams:             {df_clean['HomeTeam'].nunique()}")
    logger.info(f"Columns:           {len(df_clean.columns)}")
    
    # Result distribution
    result_counts = df_clean['FTR'].value_counts()
    logger.info(f"\nResult distribution:")
    logger.info(f"  Home wins:       {result_counts.get('H', 0)} ({result_counts.get('H', 0)/len(df_clean)*100:.1f}%)")
    logger.info(f"  Draws:           {result_counts.get('D', 0)} ({result_counts.get('D', 0)/len(df_clean)*100:.1f}%)")
    logger.info(f"  Away wins:       {result_counts.get('A', 0)} ({result_counts.get('A', 0)/len(df_clean)*100:.1f}%)")
    
    logger.info("\n✅ Data download complete!")


if __name__ == '__main__':
    main()
