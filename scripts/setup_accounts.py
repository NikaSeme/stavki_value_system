#!/usr/bin/env python3
"""
Setup Betting Accounts
Generating config/accounts.json securely
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.account_manager import AccountManager

def main():
    print("Setting up Betting Accounts Config...")
    config_path = Path("config/accounts.json")
    
    if config_path.exists():
        print(f"Config already exists at {config_path}")
        print("Skipping generation to avoid overwrite.")
        return

    manager = AccountManager(config_path=str(config_path))
    # content is auto-created by init if missing, 
    # but we want to ensure it has the structure we want.
    # actually manager.__init__ calls self._create_demo_config() if missing.
    # So just initializing it prints "Created demo config..."
    
    print("✓ Accounts setup complete.")

if __name__ == "__main__":
    main()
