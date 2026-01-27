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
        help='Sport key (e.g., soccer_epl). If omitted, runs all active leagues from config.'
    )
    parser.add_argument(
        '--regions',
        default='eu,uk,us',
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
    parser.add_argument(
        '--track-lines',
        action='store_true',
        help='Persist odds snapshots for line movement analysis (M04)'
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
    
    # If --sport passed explicitly, use it. Otherwise, load active leagues from config.
    target_leagues = []
    
    # Load league config
    try:
        import yaml
        with open('config/leagues.yaml', 'r') as f:
            league_config = yaml.safe_load(f)
    except Exception as e:
        print(f"⚠ Warning: Could not load config/leagues.yaml: {e}")
        league_config = {}

    if args.sport:
        # Single mode override
        target_leagues.append({'key': args.sport, 'group': 'manual'})
    else:
        # Multi-league mode
        print("🌍 Multi-League Mode Active")
        soccer = league_config.get('soccer', [])
        basket = league_config.get('basketball', [])
        
        for l in soccer:
            if l.get('active'):
                target_leagues.append({'key': l['key'], 'group': 'soccer'})
        for l in basket:
            if l.get('active'):
                target_leagues.append({'key': l['key'], 'group': 'basketball'})

    if not target_leagues:
        print("❌ No active leagues found in config/leagues.yaml and no --sport provided.")
        sys.exit(1)

    all_events = []
    raw_files = []

    for item in target_leagues:
        sport_key = item['key']
        print(f"\n────────────────────────────────────────")
        print(f"📊 Fetching: {sport_key}")
        
        try:
            # Respect hours_ahead
            now = datetime.now(timezone.utc)
            to_time = now + timedelta(hours=args.hours_ahead)
            
            events = fetch_odds(
                sport_key=sport_key,
                regions=args.regions,
                markets=args.markets,
                odds_format=args.odds_format,
                commence_time_from=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                commence_time_to=to_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                cfg=api_config
            )
            
            # Tag events with extra metadata for processing
            for ev in events:
                ev['_sport_group'] = item['group']
            
            all_events.extend(events)
            
            # Per-sport raw dump (optional, but good for debug)
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # raw_file = output_dir / f"raw_{sport_key}_{timestamp}.json"
            # with open(raw_file, 'w') as f:
            #    json.dump(events, f, indent=2)
            
            print(f"   ✓ Extracted {len(events)} events")
            
        except Exception as e:
            print(f"   ❌ Error fetching {sport_key}: {e}")
            continue

    # Master Raw Dump
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_master_file = output_dir / f"raw_master_{timestamp}.json"
    
    with open(raw_master_file, 'w') as f:
        json.dump(all_events, f, indent=2)
    print(f"\n💾 Saved master raw snapshot: {raw_master_file}")

    # Normalize ALL
    print(f"\n🔄 Normalizing {len(all_events)} events across all leagues...")
    
    # We use standard normalization but now we can handle basic sport tagging if needed
    # The normalization function handles standard schema.
    rows = normalize_odds_events(all_events)
    best_rows = best_price_by_outcome(rows)
    
    # Save Unified CSV
    # Instead of 'normalized_SPORT_timestamp.csv', we use 'events_latest.csv' or timestamped master
    normalized_file = output_dir / f"events_latest_{timestamp}.csv"
    
    # Also save as Parquet for V3.3 spec?
    # Spec says: data/processed/events_latest.parquet
    # We will save CSV here as standard output, and optionally parquet if pandas available
    
    if best_rows:
        import csv
        
        fieldnames = ['event_id', 'sport_key', 'commence_time', 'home_team', 'away_team', 
                     'bookmaker_key', 'bookmaker_title', 'last_update', 
                     'market_key', 'outcome_name', 'outcome_price', 'odds_snapshot_time']
        
        with open(normalized_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(best_rows)
            
        print(f"✓ Saved unified CSV: {normalized_file}")
        
        # Parquet Export (V3.3 Spec)
        try:
            import pandas as pd
            df = pd.DataFrame(best_rows)
            parquet_path = Path("data/processed/events_latest.parquet")
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(parquet_path, index=False)
            print(f"✓ Saved unified Parquet: {parquet_path}")

            # Generate Universe Summary (Step 1.3)
            universe_summary = {
                "timestamp": timestamp,
                "total_events": len(all_events),
                "by_league": df['sport_key'].value_counts().to_dict(),
                "by_bookmaker": df['bookmaker_key'].value_counts().to_dict()
            }
            summary_path = Path("audit_pack/A9_live/universe_summary.json")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, 'w') as f:
                json.dump(universe_summary, f, indent=2)
            print(f"✓ Generated Universe Summary: {summary_path}")
            
        except ImportError:
            print("⚠ Pandas/PyArrow not found. Skipping Parquet save.")

        # Line Movement Tracking (M04)
        if args.track_lines:
            print(f"\n📈 Tracking Line Movement...")
            from src.data.odds_tracker import OddsTracker
            tracker = OddsTracker()
            count_tracked = 0
            
            for event in all_events:
                try:
                    eid = event['id']
                    
                    # Extract H2H odds per bookmaker
                    odds_data = {}
                    for bk in event.get('bookmakers', []):
                        bk_key = bk['key']
                        # Find h2h market
                        h2h_market = next((m for m in bk.get('markets', []) if m['key'] == 'h2h'), None)
                        if h2h_market:
                            outcomes = {}
                            for o in h2h_market['outcomes']:
                                outcomes[o['name']] = o['price']
                            odds_data[bk_key] = outcomes
                    
                    if odds_data:
                        # Auto-detect opening line
                        is_opening = False
                        if not tracker.get_opening_odds(eid):
                            is_opening = True
                            
                        tracker.store_odds_snapshot(eid, odds_data, is_opening=is_opening)
                        count_tracked += 1
                        
                except Exception as e:
                    # Log but continue
                    continue
            
            print(f"   ✓ Tracked lines for {count_tracked} events")

    else:
        print("⚠ No events found across any league.")


if __name__ == "__main__":
    main()
