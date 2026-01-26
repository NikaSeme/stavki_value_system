"""
Telegram notification module for value bet alerts.

Sends formatted messages to Telegram when value bets are found.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

try:
    import requests
except ImportError:
    requests = None


def format_value_message(bets: List[Dict[str, Any]], top_n: int = 5, build_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Format value bets into a fixed-width table Telegram message (V5 Spec).
    """
    lines = []
    
    # 1. Build Stamp
    if build_data:
        # e.g. Build: abc1234 (Ensemble v1.0) - 2026-01-25 22:00 UTC - Fallback: None
        commit = build_data.get('commit', 'unknown')[:7]
        model_ver = build_data.get('model_version', 'v3.2')
        timestamp = build_data.get('timestamp', 'Unknown Time')
        fallback = "None" # Enforced by V5 policy
        
        lines.append(f"Build: `{commit}` ({model_ver}) - {timestamp} - Fallback: {fallback}")
        lines.append("")
    
    if not bets:
        lines.append("No value bets found.")
        return "\n".join(lines)
    
    # 2. Table Header
    # Match (League) | Market | Pick | Mod% | Imp% | Odds | EV | Stake
    # Using code block for alignment
    lines.append("```")
    # Header row
    # M:Match, Mk:Market, P:Pick, O:Odds, E:EV
    # We need to condense. 
    # Match                Pick        Odds  EV   Stake
    # -------------------------------------------------
    lines.append(f"{'Match':<20} {'Pick':<15} {'Odds':<5} {'EV':<4} {'Stake'}")
    lines.append("-" * 60)
    
    for bet in bets[:top_n]:
        # Truncate match name
        match_str = f"{bet['home_team']} vs {bet['away_team']}"
        if len(match_str) > 20:
             match_str = match_str[:19] + "…"
        
        pick_str = bet['selection']
        if len(pick_str) > 15:
            pick_str = pick_str[:14] + "…"
            
        odds_str = f"{bet['odds']:.2f}"
        ev_str = f"{int(bet['ev_pct'])}%"
        stake_str = f"{int(bet['stake_pct'])}%"
        
        # Row
        lines.append(f"{match_str:<20} {pick_str:<15} {odds_str:<5} {ev_str:<4} {stake_str}")
        
    lines.append("```")
    
    if len(bets) > top_n:
        lines.append(f"_...and {len(bets) - top_n} more bets (check logs)_")
        
    return "\n".join(lines)

def send_value_alert(
    bets: List[Dict[str, Any]],
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    top_n: int = 5,
    build_data: Optional[Dict[str, Any]] = None,
    dry_run: bool = False
) -> bool:
    """
    Send value bet alert via Telegram with Deduplication (V5 Spec).
    """
    if dry_run:
        print("Dry-Run: Skipping Telegram send.")
        print(format_value_message(bets, top_n, build_data))
        return True

    # Get credentials
    if bot_token is None: bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if chat_id is None: chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("⚠️  Telegram not configured.")
        return False
        
    # V5 Deduplication: Check alerts_sent.csv (last 48h)
    # We iterate bets and only keep NEW unique ones (EventID + Selection)
    # But wait, we usually send one message with top 5.
    # We should filter the 'bets' list BEFORE formatting.
    
    sent_log = Path("audit_pack/A9_live/alerts_sent.csv")
    recently_sent = set()
    
    if sent_log.exists():
        try:
            df_sent = pd.read_csv(sent_log)
            # Filter for last 48h if 'timestamp' exists
            # For now, just load all to be safe (or last 1000 rows)
            # Key: event_id + selection
            if 'event_id' in df_sent.columns and 'selection' in df_sent.columns:
                for _, row in df_sent.iterrows():
                    key = f"{row['event_id']}_{row['selection']}"
                    recently_sent.add(key)
        except Exception as e:
            print(f"⚠️  Deduplication warning: Could not read sent log: {e}")
            
    filtered_bets = []
    skipped_count = 0
    for b in bets:
        key = f"{b['event_id']}_{b['selection']}"
        if key in recently_sent:
            skipped_count += 1
        else:
            filtered_bets.append(b)
            
    if skipped_count > 0:
        print(f"ℹ️  Skipped {skipped_count} duplicate bets (already sent).")
        
    if not filtered_bets:
        print("ℹ️  No new value bets to send (all duplicates).")
        return True
        
    # Send
    message = format_value_message(filtered_bets, top_n, build_data)
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    try:
        response = requests.post(
            url,
            json={'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown', 'disable_web_page_preview': True},
            timeout=10
        )
        if response.status_code == 200:
            print(f"✅ Telegram alert sent (Top {min(len(filtered_bets), top_n)} bets).")
            # Update alerts_sent.csv happen in caller? 
            # Ideally caller handles logging to ensure atomic 'Sent -> Log'.
            # But duplicate check happened here. We should return the filtered list?
            # Or assume caller logs ALL 'filtered_bets' as sent.
            return True # Simple boolean for now
        else:
            print(f"❌ Telegram send failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False


def is_telegram_configured() -> bool:
    """Check if Telegram credentials are configured."""
    return bool(os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'))
