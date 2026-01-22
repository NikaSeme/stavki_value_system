"""
Environment configuration loader for STAVKI.

Loads from .env, api1.env, or OS environment variables.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_env_config() -> dict:
    """
    Load environment configuration from .env, api1.env, or OS env.
    
    Priority:
    1. .env file (if exists)
    2. api1.env file (if exists)
    3. OS environment variables
    
    Returns:
        Dict with environment variables
        
    Raises:
        SystemExit if ODDS_API_KEY not found
    """
    project_root = Path(__file__).parent.parent.parent
    
    # Try .env first
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded config from: {env_path}")
    else:
        # Try api1.env
        api_env_path = project_root / "api1.env"
        if api_env_path.exists():
            load_dotenv(api_env_path)
            print(f"✓ Loaded config from: {api_env_path}")
        else:
            print("ℹ Using OS environment variables (no .env or api1.env found)")
    
    # Get configuration
    config = {
        'ODDS_API_KEY': os.getenv('ODDS_API_KEY'),
        'ODDS_API_BASE': os.getenv('ODDS_API_BASE', 'https://api.the-odds-api.com'),
        'ODDS_REGIONS': os.getenv('ODDS_REGIONS', 'eu'),
        'ODDS_MARKETS': os.getenv('ODDS_MARKETS', 'h2h'),
        'ODDS_ODDS_FORMAT': os.getenv('ODDS_ODDS_FORMAT', 'decimal'),
    }
    
    # Validate required fields
    if not config['ODDS_API_KEY']:
        print("\n❌ ERROR: ODDS_API_KEY not found!")
        print("\nPlease set it in one of:")
        print(f"  1. {env_path} (create from .env.example)")
        print(f"  2. {api_env_path}")
        print("  3. OS environment: export ODDS_API_KEY=your_key_here")
        print("\nGet your free API key at: https://the-odds-api.com")
        sys.exit(1)
    
    # Never print the key
    print(f"✓ ODDS_API_KEY: {'*' * 8}{config['ODDS_API_KEY'][-4:]}")
    print(f"✓ API Base: {config['ODDS_API_BASE']}")
    
    return config


def get_odds_api_key() -> str:
    """
    Get ODDS_API_KEY from config.
    
    Returns:
        API key string
        
    Raises:
        SystemExit if not found
    """
    config = load_env_config()
    return config['ODDS_API_KEY']
