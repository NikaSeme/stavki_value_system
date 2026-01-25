"""
Telegram notification module for value bet alerts.

Sends formatted messages to Telegram when value bets are found.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None


def format_value_message(bets: List[Dict[str, Any]], top_n: int = 5) -> str:
    """
    Format value bets into a concise Telegram message.
    
    Args:
        bets: List of value bet dictionaries
        top_n: Number of bets to include in message
        
    Returns:
        Formatted message string
    """
    if not bets:
        return "🔍 No value bets found in latest odds."
    
    lines = ["🎯 **VALUE BETS FOUND**\n"]
    
    # Show top N bets
    for i, bet in enumerate(bets[:top_n], 1):
        lines.append(f"**{i}. {bet['selection']}** @ {bet['odds']}")
        lines.append(f"   {bet['home_team']} vs {bet['away_team']}")
        lines.append(f"   EV: +{bet['ev_pct']:.1f}% | {bet['bookmaker']}")
        lines.append(f"   Model: {bet['p_model']*100:.1f}% | Implied: {bet['p_implied']*100:.1f}%")
        lines.append(f"   Recommended Stake: £{bet['stake']:.2f}\n")
    
    if len(bets) > top_n:
        lines.append(f"_...and {len(bets) - top_n} more bets_")
    
    # Summary stats
    avg_ev = sum(b['ev'] for b in bets) / len(bets)
    lines.append(f"\n📊 **Summary**")
    lines.append(f"Total bets: {len(bets)}")
    lines.append(f"Avg EV: +{avg_ev*100:.2f}%")
    lines.append(f"Best EV: +{bets[0]['ev_pct']:.1f}%")
    
    return "\n".join(lines)


def send_value_alert(
    bets: List[Dict[str, Any]],
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    top_n: int = 5
) -> bool:
    """
    Send value bet alert via Telegram.
    
    Args:
        bets: List of value bets
        bot_token: Telegram bot token (or None to use env var)
        chat_id: Telegram chat ID (or None to use env var)
        top_n: Number of bets to include
        
    Returns:
        True if sent successfully, False otherwise
    """
    # Get credentials from env if not provided
    if bot_token is None:
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if chat_id is None:
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    # Validate
    if not bot_token or not chat_id:
        print("⚠️  Telegram not configured (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")
        return False
    
    if requests is None:
        print("⚠️  requests library not available, cannot send Telegram message")
        return False
    
    # Format message
    message = format_value_message(bets, top_n)
    
    # Send via Telegram Bot API
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    try:
        response = requests.post(
            url,
            json={
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"✅ Telegram alert sent to chat {chat_id}")
            return True
        else:
            print(f"❌ Telegram send failed: {response.status_code} {response.text[:100]}")
            return False
            
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False


def is_telegram_configured() -> bool:
    """Check if Telegram credentials are configured."""
    return bool(os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'))
