"""
Telegram bot integration for sending alerts.

Sends real-time notifications for:
- Value bet opportunities
- Line movements
- Performance warnings
- System health issues
"""

import os
import logging
from typing import Optional
from telegram import Bot
from telegram.error import TelegramError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelegramAlertBot:
    """Send alerts via Telegram bot."""
    
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Initialize Telegram bot.
        
        Args:
            token: Bot token (reads from TELEGRAM_BOT_TOKEN env if None)
            chat_id: Chat ID (reads from TELEGRAM_CHAT_ID env if None)
        """
        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in environment")
        
        if not self.chat_id:
            logger.warning("TELEGRAM_CHAT_ID not set - alerts will fail")
        
        self.bot = Bot(token=self.token)
        logger.info("✓ Telegram bot initialized")
    
    async def send_alert(
        self,
        message: str,
        parse_mode: str = 'Markdown',
        disable_notification: bool = False
    ) -> bool:
        """
        Send alert message.
        
        Args:
            message: Alert text
            parse_mode: 'Markdown' or 'HTML'
            disable_notification: Silent notification
            
        Returns:
            True if sent successfully
        """
        if not self.chat_id:
            logger.error("Cannot send alert - no chat_id configured")
            return False
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode,
                disable_notification=disable_notification
            )
            logger.info("✓ Alert sent via Telegram")
            return True
            
        except TelegramError as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False
    
    def send_alert_sync(self, message: str, **kwargs) -> bool:
        """Synchronous wrapper for send_alert."""
        import asyncio
        try:
            return asyncio.run(self.send_alert(message, **kwargs))
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test bot connection."""
        try:
            me = await self.bot.get_me()
            logger.info(f"✓ Connected to bot: @{me.username}")
            return True
        except TelegramError as e:
            logger.error(f"Connection test failed: {e}")
            return False


def test_telegram_bot():
    """Test Telegram bot integration."""
    print("=" * 60)
    print("TELEGRAM BOT TEST")
    print("=" * 60)
    
    try:
        import asyncio
        
        bot = TelegramAlertBot()
        
        # Test connection
        print("\n1. Testing connection...")
        connected = asyncio.run(bot.test_connection())
        
        if not connected:
            print("❌ Connection failed - check token")
            return
        
        # Send test message
        print("\n2. Sending test alert...")
        message = """
🧪 **TEST ALERT**

Telegram integration working!

System: Ready ✓
Alerts: Enabled ✓
        """
        
        success = bot.send_alert_sync(message)
        
        if success:
            print("✅ Test alert sent - check Telegram")
        else:
            print("❌ Alert failed - check chat_id")
    
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == '__main__':
    test_telegram_bot()
