#!/usr/bin/env python3
"""
Odds API Debug Dump (Audit v3)
Fetches raw odds for 1 sport to verify connectivity, json structure, and parsing.
"""
import sys
import json
import csv
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.odds_api_client import fetch_odds, load_config_from_env
from src.data.odds_normalize import normalize_odds_events

OUTPUT_DIR = Path("audit_pack/A4_odds_integrity")
RAW_FILE = OUTPUT_DIR / "odds_raw_sample.json"
PARSED_FILE = OUTPUT_DIR / "odds_parsed_sample.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=== Odds API Debug Dump ===")
    
    try:
        cfg = load_config_from_env()
        # Mask key for log safety
        print(f"API Key found: ...{cfg.api_key[-4:] if cfg.api_key else 'None'}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)
        
    # Fetch
    print("Fetching soccer_epl odds (eu, h2h)...")
    try:
        # We fetch minimal set
        events = fetch_odds(
            sport_key="soccer_epl", 
            regions="eu", 
            markets="h2h",
            cfg=cfg
        )
    except Exception as e:
        print(f"❌ Fetch failed: {e}")
        # Create empty error file to fail audit nicely but with trace
        with open(RAW_FILE, "w") as f:
            json.dump({"error": str(e)}, f)
        sys.exit(1)
        
    # Validations on raw
    print(f"Fetched {len(events)} events.")
    
    with open(RAW_FILE, "w") as f:
        json.dump(events, f, indent=2)
    print(f"Saved raw to {RAW_FILE}")
    
    if not events:
        print("⚠ No events returned (Season break?).")
        sys.exit(0)
        
    # Parsing
    rows = normalize_odds_events(events)
    print(f"Normalized to {len(rows)} rows.")
    
    # Save parsed sample
    if rows:
        fieldnames = ["bookmaker_key", "market_key", "outcome_name", "outcome_price", "commence_time"]
        with open(PARSED_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for r in rows[:50]: # Save top 50
                writer.writerow(r)
        print(f"Saved parsed sample to {PARSED_FILE}")
    else:
        print("⚠ No rows after normalization.")

    print("✅ Odds API Debug Success")

if __name__ == "__main__":
    main()
