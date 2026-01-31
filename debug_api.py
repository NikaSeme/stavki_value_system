
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src.config.env import load_env_config
from src.data.odds_api_client import OddsAPIConfig, fetch_odds
from datetime import datetime, timezone

config = load_env_config()
api = OddsAPIConfig(api_key=config['ODDS_API_KEY'], base_url=config['ODDS_API_BASE'])

print(f"API Key: {config['ODDS_API_KEY'][:5]}...")

# Try fetching EPL
try:
    events = fetch_odds(
        sport_key='soccer_epl',
        regions='eu',
        markets='h2h',
        odds_format='decimal',
        commence_time_from=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        cfg=api
    )
    print(f"EPL Events Found: {len(events)}")
    if events:
        print(f"Sample Event: {events[0]['home_team']} vs {events[0]['away_team']}")
except Exception as e:
    print(f"Error fetching EPL: {e}")
