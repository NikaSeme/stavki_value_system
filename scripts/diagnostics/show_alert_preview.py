import sys
from pathlib import Path
import pandas as pd
import uuid
import subprocess

# Add project root
sys.path.insert(0, str(Path.cwd()))

from src.integration.telegram_notify import format_value_message

def get_git_revision_short_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('ascii')
    except:
        return "unknown"

def main():
    # Mock data resembling real run
    bets = [
        {
            'selection': 'Arsenal',
            'odds': 2.10,
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
            'ev_pct': 8.5,
            'bookmaker': 'Pinnacle',
            'p_model': 0.52,
            'p_implied': 0.476,
            'stake': 50.00,
            'ev': 0.085
        },
        {
            'selection': 'Over 2.5',
            'odds': 1.95,
            'home_team': 'Liverpool',
            'away_team': 'Man City',
            'ev_pct': 6.2,
            'bookmaker': 'Bet365',
            'p_model': 0.54,
            'p_implied': 0.513,
            'stake': 42.50,
            'ev': 0.062
        }
    ]
    
    commit = get_git_revision_short_hash()
    run_id = str(uuid.uuid4())[:8]
    
    build_data = {
        'commit': commit,
        'run_id': run_id,
        'pipeline': 'run_value_finder.py',
        'bookmaker_mode': 'SINGLE_BOOK',
        'models_loaded': True,
        'fallback_used': False
    }
    
    msg = format_value_message(bets, top_n=5, build_data=build_data)
    print("=== TELEGRAM ALERT PREVIEW ===")
    print(msg)
    print("==============================")

if __name__ == "__main__":
    main()
