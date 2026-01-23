"""
Daily summary report generator.

Sends daily performance summary via Telegram.
Run via cron/schedule at midnight.
"""

import sys
from pathlib import Path
from datetime import datetime, date

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.performance_monitor import PerformanceMonitor
from src.alerts.telegram_bot import TelegramAlertBot


def generate_daily_summary():
    """Generate and send daily summary."""
    print(f"Generating daily summary for {date.today()}...")
    
    # Initialize
    monitor = PerformanceMonitor()
    bot = TelegramAlertBot()
    
    # Get today's stats
    today_stats = monitor.get_performance_summary('today')
    week_stats = monitor.get_performance_summary('week')
    all_stats = monitor.get_performance_summary('all')
    
    # Format message
    message = f"""
📊 **DAILY SUMMARY** - {date.today().strftime('%Y-%m-%d')}

**Today:**
Bets: {today_stats['total_bets']}
Wins: {today_stats['wins']} ({today_stats['hit_rate']:.1f}%)
Profit: £{today_stats['total_profit']:+.2f}
ROI: {today_stats['roi']:+.1f}%

**This Week:**
Bets: {week_stats['total_bets']}
ROI: {week_stats['roi']:+.1f}%
CLV: {week_stats['clv']:+.1f}%

**All-Time:**
Total Bets: {all_stats['total_bets']}
Hit Rate: {all_stats['hit_rate']:.1f}%
ROI: {all_stats['roi']:+.1f}%
Max Drawdown: £{all_stats['max_drawdown']:.2f}
Recent: {all_stats['recent_record']}

📈 Keep tracking!
    """.strip()
    
    # Send
    success = bot.send_alert_sync(message, disable_notification=False)
    
    if success:
        print("✅ Daily summary sent successfully")
    else:
        print("❌ Failed to send daily summary")
    
    # Save snapshot
    monitor.save_snapshot()
    print("✓ Performance snapshot saved")


if __name__ == '__main__':
    generate_daily_summary()
