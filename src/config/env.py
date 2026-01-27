"""
Environment configuration loader for STAVKI.

Loads from .env, api1.env, or OS environment variables.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_env_config(env_path_override: Optional[str] = None) -> dict:
    """
    Load environment configuration.
    
    Args:
        env_path_override: Optional path to specific .env file

    Priority:
    0. env_path_override
    1. /etc/stavki/stavki.env
    2. .env file
    3. api1.env file
    4. OS environment variables
    
    Returns:
        Dict with environment variables
    """
    project_root = Path(__file__).parent.parent.parent

    # 0. Override
    if env_path_override and Path(env_path_override).exists():
        load_dotenv(env_path_override, override=True)
        # print(f"✓ Loaded config from: {env_path_override}")

    # 1. /etc/stavki/stavki.env (Production)
    elif Path("/etc/stavki/stavki.env").exists():
        load_dotenv("/etc/stavki/stavki.env", override=True)
        # print("✓ Loaded config from: /etc/stavki/stavki.env")
    
    # Try .env next
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        # print(f"✓ Loaded config from: {env_path}")
    else:
        # Try api1.env
        api_env_path = project_root / "api1.env"
        if api_env_path.exists():
            load_dotenv(api_env_path)
            # print(f"✓ Loaded config from: {api_env_path}")
    
    # Get configuration
    config = {
        'ODDS_API_KEY': os.getenv('ODDS_API_KEY'),
        'ODDS_API_BASE': os.getenv('ODDS_API_BASE', 'https://api.the-odds-api.com'),
        'ODDS_REGIONS': os.getenv('ODDS_REGIONS', 'eu'),
        'ODDS_MARKETS': os.getenv('ODDS_MARKETS', 'h2h'),
        'ODDS_ODDS_FORMAT': os.getenv('ODDS_ODDS_FORMAT', 'decimal'),
        'NEWS_API_KEY': os.getenv('NEWS_API_KEY'), # Added for Model C
    }
    
    return config

def get_odds_api_key() -> str:
    """Get ODDS_API_KEY."""
    config = load_env_config()
    if not config['ODDS_API_KEY']:
         print("❌ ODDS_API_KEY missing.")
         sys.exit(1)
    return config['ODDS_API_KEY']

# Alias for compatibility
load_env = load_env_config
