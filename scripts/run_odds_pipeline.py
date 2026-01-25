#!/usr/bin/env python3
"""
STAVKI Odds Pipeline - One-Command Entrypoint

Fetches odds from The Odds API, normalizes them, and saves outputs.

Usage:
    python run_odds_pipeline.py --sport soccer_epl --regions eu --markets h2h
    
    python run_odds_pipeline.py --help
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.env import load_env_config
from src.data.odds_api_client import OddsAPIConfig, fetch_odds
from src.data.odds_normalize import best_price_by_outcome, normalize_odds_events


def main():
    """Main entrypoint for odds pipeline."""
    parser = argparse.ArgumentParser(
        description="Fetch and normalize odds from The Odds API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_odds_pipeline.py --sport soccer_epl --regions eu --markets h2h
  python run_odds_pipeline.py --sport basketball_nba --regions us
  python run_odds_pipeline.py --sport americanfootball_nfl --output-dir myoutputs/
        """
    )
    
    parser.add_argument(
        '--sport',
        default='soccer_epl',
        help='Sport key (e.g., soccer_epl, basketball_nba, americanfootball_nfl)'
    )
    parser.add_argument(
        '--regions',
        default='eu',
        help='Regions (comma-separated: eu, us, uk, au)'
    )
    parser.add_argument(
        '--markets',
        default='h2h',
        help='Markets (comma-separated: h2h, spreads, totals)'
    )
    parser.add_argument(
        '--odds-format',
        default='decimal',
        help='Odds format: decimal, american'
    )
    parser.add_argument(
        '--output-dir',
        default='outputs/odds',
        help='Output directory for results'
    )
    parser.add_argument(
        '--hours-ahead',
        type=int,
        default=48,
        help='Fetch events starting within N hours from now'
    )
    
    args = parser.parse_args()
    
    # Load environment config
    print("=" * 70)
    print("STAVKI ODDS PIPELINE")
    print("=" * 70)
    
    try:
        env_config = load_env_config()
    except SystemExit:
        return  # load_env_config already printed error
    
    # Create API config
    api_config = OddsAPIConfig(
        api_key=env_config['ODDS_API_KEY'],
        base_url=env_config['ODDS_API_BASE']
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch odds
    print(f"\n📊 Fetching odds...")
    print(f"  Sport:        {args.sport}")
    print(f"  Regions:      {args.regions}")
    print(f"  Markets:      {args.markets}")
    print(f"  Odds Format:  {args.odds_format}")
    print(f"  Time Range:   Next {args.hours_ahead} hours")
    
    now = datetime.now(timezone.utc)
    to_time = now + timedelta(hours=args.hours_ahead)
    
    try:
        events = fetch_odds(
            sport_key=args.sport,
            regions=args.regions,
            markets=args.markets,
            odds_format=args.odds_format,
            commence_time_from=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            commence_time_to=to_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            cfg=api_config
        )
    except Exception as e:
        print(f"\n❌ Error fetching odds: {e}")
        sys.exit(1)
    
    # Save raw JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_file = output_dir / f"raw_{args.sport}_{timestamp}.json"
    
    with open(raw_file, 'w') as f:
        json.dump(events, f, indent=2)
    
    print(f"\n✓ Saved raw data: {raw_file}")
    
    # Normalize odds
    print(f"\n🔄 Normalizing odds...")
    rows = normalize_odds_events(events)
    best_rows = best_price_by_outcome(rows)
    
    # Save normalized CSV
    normalized_file = output_dir / f"normalized_{args.sport}_{timestamp}.csv"
    
    if best_rows:
        import csv
        
        fieldnames = ['event_id', 'sport_key', 'commence_time', 'home_team', 'away_team', 
                     'bookmaker_key', 'bookmaker_title', 'last_update', 
                     'market_key', 'outcome_name', 'outcome_price', 'odds_snapshot_time']
        
        with open(normalized_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(best_rows)
        
        print(f"✓ Saved normalized data: {normalized_file}")
    else:
        print(f"⚠ No odds data to normalize")
    
    # Print summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Events fetched:       {len(events)}")
    print(f"Normalized rows:      {len(rows)}")
    print(f"Best prices:          {len(best_rows)}")
    
    if events:
        # Extract bookmakers
        bookmakers = set()
        for event in events:
            for bookmaker in event.get('bookmakers', []):
                bookmakers.add(bookmaker.get('key', 'unknown'))
        
        print(f"Bookmakers:           {len(bookmakers)}")
        if bookmakers:
            print(f"  {', '.join(sorted(bookmakers))}")
    
    print("=" * 70)
    print(f"\n✅ Pipeline complete!")
    print(f"📁 Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
