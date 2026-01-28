#!/usr/bin/env python3
"""
Download historical data for multiple leagues from Football-Data.co.uk

Supports: Bundesliga, La Liga, Serie A, Champions League
Seasons: 2021-22, 2022-23, 2023-24

Usage:
    python download_multi_league_data.py --leagues bundesliga laliga
    python download_multi_league_data.py --all
"""

import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

# League codes from Football-Data.co.uk
LEAGUE_CODES = {
    'bundesliga': {
        'code': 'D1',
        'name': 'Bundesliga',
        'country': 'Germany'
    },
    'laliga': {
        'code': 'SP1',
        'name': 'La Liga',
        'country': 'Spain'
    },
    'seriea': {
        'code': 'I1',
        'name': 'Serie A',
        'country': 'Italy'
    },
    'ligue1': {
        'code': 'F1',
        'name': 'Ligue 1',
        'country': 'France'
    }
    # 'champions_league': {
    #     'code': 'EC1',
    #     'name': 'Champions League',
    #     'country': 'Europe'
    # }
}

# Season mappings (Football-Data uses YY format)
SEASONS = {
    '2021-22': '2122',
    '2022-23': '2223',
    '2023-24': '2324'
}

def download_league_season(league_key, season_key, output_dir):
    """
    Download a single league's season data.
    
    Args:
        league_key: Key from LEAGUE_CODES dict
        season_key: Key from SEASONS dict (e.g., '2021-22')
        output_dir: Directory to save CSV
    
    Returns:
        Path to downloaded file, or None if failed
    """
    league_info = LEAGUE_CODES[league_key]
    season_code = SEASONS[season_key]
    league_code = league_info['code']
    
    # URL format: https://www.football-data.co.uk/mmz4281/2122/D1.csv
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"
    
    print(f"  Downloading {league_info['name']} {season_key}...")
    print(f"    URL: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save raw CSV
        output_file = output_dir / f"{league_key}_{season_key.replace('-', '_')}.csv"
        output_file.write_bytes(response.content)
        
        # Validate it's actually CSV data
        try:
            df = pd.read_csv(output_file)
            print(f"    ✓ Downloaded {len(df)} matches")
            return output_file
        except Exception as e:
            print(f"    ❌ Invalid CSV: {e}")
            output_file.unlink()
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"    ❌ Download failed: {e}")
        return None

def standardize_csv(raw_file, league_key, season):
    """
    Standardize Football-Data.co.uk CSV to STAVKI format.
    
    Expected columns:
        Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR
    
    Args:
        raw_file: Path to raw CSV
        league_key: League identifier
        season: Season string (e.g., '2021-22')
    
    Returns:
        DataFrame with standardized columns
    """
    df = pd.read_csv(raw_file, encoding='latin1')
    
    # Check required columns exist
    required = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Select and rename
    standardized = df[required].copy()
    
    # Add metadata
    standardized['Season'] = season
    standardized['League'] = league_key
    
    # Parse dates
    standardized['Date'] = pd.to_datetime(standardized['Date'], format='%d/%m/%Y', errors='coerce')
    
    # Clean team names (strip whitespace)
    standardized['HomeTeam'] = standardized['HomeTeam'].str.strip()
    standardized['AwayTeam'] = standardized['AwayTeam'].str.strip()
    
    # Validate FTR
    valid_results = standardized['FTR'].isin(['H', 'D', 'A'])
    if not valid_results.all():
        print(f"    ⚠️ Warning: {(~valid_results).sum()} invalid FTR values")
        standardized = standardized[valid_results]
    
    # Remove duplicates
    before = len(standardized)
    standardized = standardized.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'])
    if len(standardized) < before:
        print(f"    ⚠️ Removed {before - len(standardized)} duplicate matches")
    
    return standardized

def main():
    parser = argparse.ArgumentParser(description='Download multi-league historical data')
    parser.add_argument('--leagues', nargs='+', choices=list(LEAGUE_CODES.keys()),
                       help='Leagues to download')
    parser.add_argument('--all', action='store_true',
                       help='Download all available leagues')
    parser.add_argument('--seasons', nargs='+', choices=list(SEASONS.keys()),
                       default=list(SEASONS.keys()),
                       help='Seasons to download (default: all)')
    
    args = parser.parse_args()
    
    # Determine which leagues to download
    if args.all:
        leagues_to_download = list(LEAGUE_CODES.keys())
    elif args.leagues:
        leagues_to_download = args.leagues
    else:
        print("❌ Error: Specify --leagues or --all")
        return 1
    
    print("=" * 70)
    print("MULTI-LEAGUE DATA DOWNLOADER")
    print("=" * 70)
    print(f"Leagues: {', '.join(leagues_to_download)}")
    print(f"Seasons: {', '.join(args.seasons)}")
    print()
    
    # Setup directories
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / 'data' / 'raw'
    processed_dir = base_dir / 'data' / 'processed'
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Download each league
    all_standardized = []
    
    for league_key in leagues_to_download:
        print(f"\n📥 {LEAGUE_CODES[league_key]['name']}")
        print("-" * 70)
        
        league_raw_dir = raw_dir / league_key
        league_raw_dir.mkdir(exist_ok=True)
        
        league_data = []
        
        for season in args.seasons:
            raw_file = download_league_season(league_key, season, league_raw_dir)
            
            if raw_file:
                try:
                    standardized = standardize_csv(raw_file, league_key, season)
                    league_data.append(standardized)
                    all_standardized.append(standardized)
                    print(f"    ✓ Processed {len(standardized)} valid matches")
                except Exception as e:
                    print(f"    ❌ Processing failed: {e}")
            
            # Be polite to the server
            time.sleep(1)
        
        # Save league-specific combined file
        if league_data:
            combined = pd.concat(league_data, ignore_index=True)
            combined = combined.sort_values('Date').reset_index(drop=True)
            
            output_file = processed_dir / f"{league_key}_historical_2021_2024.csv"
            combined.to_csv(output_file, index=False)
            print(f"\n  ✅ Saved: {output_file}")
            print(f"     Total: {len(combined)} matches across {len(league_data)} seasons")
    
    # Create multi-league combined file
    if all_standardized:
        print("\n" + "=" * 70)
        print("COMBINING ALL LEAGUES")
        print("=" * 70)
        
        multi_league = pd.concat(all_standardized, ignore_index=True)
        multi_league = multi_league.sort_values('Date').reset_index(drop=True)
        
        output_file = processed_dir / 'multi_league_combined_2021_2024.csv'
        multi_league.to_csv(output_file, index=False)
        
        print(f"\n✅ Multi-League Dataset Created!")
        print(f"   File: {output_file}")
        print(f"   Total Matches: {len(multi_league)}")
        print(f"   Leagues: {multi_league['League'].nunique()}")
        print(f"   Date Range: {multi_league['Date'].min()} to {multi_league['Date'].max()}")
        print()
        print("League Breakdown:")
        for league, count in multi_league['League'].value_counts().items():
            print(f"  - {LEAGUE_CODES[league]['name']}: {count} matches")
    
    print("\n🎉 Download complete!")
    return 0

if __name__ == '__main__':
    exit(main())
