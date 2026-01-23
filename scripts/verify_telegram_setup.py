#!/usr/bin/env python3
"""
Telegram Bot Setup Verification Script

Checks if Telegram bot is properly configured and working.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_env_file():
    """Check if .env file exists."""
    env_path = Path('.env')
    if not env_path.exists():
        print("❌ .env file not found")
        print("\n📝 Create .env file with:")
        print("   TELEGRAM_BOT_TOKEN=your_bot_token")
        print("   TELEGRAM_CHAT_ID=your_chat_id")
        return False
    print("✓ .env file exists")
    return True

def check_env_variables():
    """Check if required environment variables are set."""
    from dotenv import load_dotenv
    load_dotenv()
    
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    issues = []
    
    if not token:
        issues.append("TELEGRAM_BOT_TOKEN not set")
    elif token == 'your_bot_token_here':
        issues.append("TELEGRAM_BOT_TOKEN still has placeholder value")
    else:
        print(f"✓ TELEGRAM_BOT_TOKEN set (length: {len(token)})")
    
    if not chat_id:
        issues.append("TELEGRAM_CHAT_ID not set")
    elif chat_id == 'your_chat_id_here':
        issues.append("TELEGRAM_CHAT_ID still has placeholder value")
    else:
        print(f"✓ TELEGRAM_CHAT_ID set ({chat_id})")
    
    if issues:
        print("\n❌ Environment variable issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    return True

def check_gitignore():
    """Check if .env is in .gitignore."""
    gitignore_path = Path('.gitignore')
    if not gitignore_path.exists():
        print("⚠️  .gitignore not found")
        return False
    
    with open(gitignore_path) as f:
        content = f.read()
    
    if '.env' in content:
        print("✓ .env is in .gitignore (safe from Git)")
        return True
    else:
        print("❌ .env NOT in .gitignore - SECURITY RISK!")
        print("   Run: echo '.env' >> .gitignore")
        return False

def test_bot_connection():
    """Test actual connection to Telegram bot."""
    try:
        from src.alerts.telegram_bot import TelegramAlertBot
        
        print("\n🔌 Testing Telegram connection...")
        bot = TelegramAlertBot()
        
        # Try to send test message
        success = bot.send_alert_sync("✅ Setup verification test - your bot is working!")
        
        if success:
            print("✅ SUCCESS! Check your Telegram for test message")
            return True
        else:
            print("❌ Failed to send message - check your credentials")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def main():
    """Run all verification checks."""
    print("="*70)
    print("TELEGRAM BOT SETUP VERIFICATION")
    print("="*70)
    print()
    
    checks_passed = 0
    total_checks = 4
    
    # Check 1: .env file exists
    if check_env_file():
        checks_passed += 1
    print()
    
    # Check 2: Environment variables set
    if check_env_variables():
        checks_passed += 1
    print()
    
    # Check 3: .gitignore configured
    if check_gitignore():
        checks_passed += 1
    print()
    
    # Check 4: Bot connection works
    if checks_passed == 3:  # Only test if previous checks passed
        if test_bot_connection():
            checks_passed += 1
    else:
        print("⏭️  Skipping connection test (fix above issues first)")
    
    print()
    print("="*70)
    print(f"RESULT: {checks_passed}/{total_checks} checks passed")
    print("="*70)
    
    if checks_passed == total_checks:
        print("\n🎉 All checks passed! Your Telegram bot is ready to use.")
        print("\n📖 Next steps:")
        print("   1. Test alerts: python src/alerts/alert_manager.py")
        print("   2. Integrate with your betting system")
    else:
        print("\n⚠️  Please fix the issues above and run this script again.")
        print("\n📖 Setup guide: docs/TELEGRAM_BOT_SETUP.md")
    
    return checks_passed == total_checks

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
