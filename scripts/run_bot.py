#!/usr/bin/env python3
"""
Run STAVKI Telegram Bot.

Usage:
    python scripts/run_bot.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.bot.telegram_bot import StavkiBot
from src.logging_setup import get_logger, setup_logging

# Load environment
load_dotenv()

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)


def main():
    """Run the bot."""
    # Get configuration
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    allowed_users_str = os.getenv('TELEGRAM_ALLOWED_USERS', '')
    
    if not token:
        print("❌ Error: TELEGRAM_BOT_TOKEN not set in .env")
        print("Add to .env:")
        print("TELEGRAM_BOT_TOKEN=your_token_here")
        sys.exit(1)
    
    if not allowed_users_str:
        print("⚠️  Warning: TELEGRAM_ALLOWED_USERS not set")
        print("Anyone can access the bot!")
        allowed_users = []
    else:
        try:
            allowed_users = [
                int(uid.strip()) 
                for uid in allowed_users_str.split(',') 
                if uid.strip()
            ]
        except ValueError as e:
            print(f"❌ Error parsing TELEGRAM_ALLOWED_USERS: {e}")
            sys.exit(1)
    
    # Create and run bot
    try:
        bot = StavkiBot(token, allowed_users)
        bot.run()
    except KeyboardInterrupt:
        print("\n\n👋 Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}", exc_info=True)
        print(f"\n❌ Bot crashed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
