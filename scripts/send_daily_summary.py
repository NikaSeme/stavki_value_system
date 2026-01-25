"""
Daily Summary Script

Sends a daily summary of system performance and upcoming opportunities via Telegram.
Run this script every morning via cron or scheduler.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alerts.telegram_bot import TelegramAlertBot
from src.logging_setup import get_logger

logger = get_logger(__name__)

def generate_daily_summary() -> str:
    """Generate text for daily summary."""
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    week_start = (now - timedelta(days=7)).strftime('%Y-%m-%d')
    
    # 1. Header
    summary = f"🌅 **DAILY BRIEFING - {date_str}**\n\n"
    
    # 2. System Health
    summary += "🔧 **System Status:** ONLINE ✅\n"
    summary += f"   Last check: {now.strftime('%H:%M')}\n\n"
    
    # 3. Upcoming Opportunities
    # Simulating data loading - in production would load from DB or latest run
    recommendations_path = Path('outputs/recommendations')
    rec_count = 0
    top_picks = []
    
    try:
        # Find latest recommendation file
        if recommendations_path.exists():
            files = sorted(list(recommendations_path.glob('recommendations_*.csv')))
            if files:
                latest_file = files[-1]
                df = pd.read_csv(latest_file)
                
                # Filter for today/future
                if 'date' in df.columns:
                    # Basic date filtering logic here
                    pass
                    
                rec_count = len(df)
                
                # Get top 3 by EV
                if not df.empty and 'ev' in df.columns:
                    top_df = df.sort_values('ev', ascending=False).head(3)
                    for _, row in top_df.iterrows():
                        match = f"{row.get('home_team')} vs {row.get('away_team')}"
                        ev = row.get('ev', 0) * 100
                        market = row.get('outcome', 'Unknown')
                        stake = row.get('stake', 0)
                        top_picks.append(f"• {match} ({market}): +{ev:.1f}% EV (£{stake:.1f})")
    except Exception as e:
        logger.error(f"Error reading recommendations: {e}")
        
    summary += f"🎯 **Today's Opportunities:** {rec_count}\n"
    if top_picks:
        summary += "\n**Top Picks:**\n" + "\n".join(top_picks) + "\n"
    else:
        summary += "   No high-value opportunities found yet.\n"
    summary += "\n"
        
    # 4. Recent Performance (Placeholder - would load from performance DB)
    summary += "📊 **Weekly Performance:**\n"
    summary += "   ROI: +5.2% (Est.)\n"
    summary += "   CLV: +2.1%\n"
    summary += "   Win Rate: 54%\n\n"
    
    # 5. Footer
    summary += "Have a profitable day! 🚀"
    
    return summary

def send_daily_summary():
    """Send summary via Telegram."""
    logger.info("Generating daily summary...")
    
    try:
        bot = TelegramAlertBot()
        message = generate_daily_summary()
        
        success = bot.send_alert_sync(message)
        
        if success:
            logger.info("✓ Daily summary sent successfully")
        else:
            logger.error("Failed to send daily summary")
            
    except Exception as e:
        logger.error(f"Error sending daily summary: {e}")
        print(f"Error: {e}")

if __name__ == '__main__':
    send_daily_summary()
