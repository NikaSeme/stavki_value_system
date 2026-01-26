#!/usr/bin/env python3
"""
Stavki Telegram Command Bot (Interactive Layer)

Runs a polling bot to accept commands:
  /now     - Trigger pipeline immediately
  /top     - Show top bets from last run
  /status  - Show system status and version
  /dryrun  - Run simulation (no alerts)
  /help    - Show commands

Security:
  - Whitelist: Only chat_ids in TELEGRAM_ADMIN_IDS (csv) or TELEGRAM_CHAT_ID allowed.
  - Rate Limit: /now max once every 5 mins.
  - Lock: Prevents concurrent runs.

Usage:
  python scripts/run_telegram_bot.py
"""

import os
import sys
import time
import subprocess
import logging
import csv
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from telegram.constants import ParseMode

from src.config.env import load_env_config

# Setup Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Global Lock
PIPELINE_LOCK = False
LAST_RUN_TIME = None
RUN_COOLDOWN_SECONDS = 300 # 5 mins

def check_auth(update: Update) -> bool:
    """Check if user/chat is authorized."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    # Load allowed IDs from env
    # TELEGRAM_ADMIN_IDS can be comma-separated list of IDs
    allowed_ids = os.getenv("TELEGRAM_ADMIN_IDS", "")
    target_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    
    valid_ids = []
    if allowed_ids:
        valid_ids.extend([x.strip() for x in allowed_ids.split(',') if x.strip()])
    if target_chat_id:
        valid_ids.append(target_chat_id)
        
    s_uid = str(user_id)
    s_cid = str(chat_id)
    
    # Allow if user_id in whitelist OR chat_id matches target channel/group
    is_allowed = (s_uid in valid_ids) or (s_cid in valid_ids)
    
    if not is_allowed:
        logger.warning(f"Unauthorized access attempt from User {user_id} in Chat {chat_id}")
        return False
    return True

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not check_auth(update): return
    await update.message.reply_text("🤖 Stavki V5 Bot Online.\nUse /help to see commands.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not check_auth(update): return
    msg = (
        "📜 *Stavki Bot Commands*\n\n"
        "`/now` - Run pipeline immediately\n"
        "`/top [N]` - Show top N bets from last run\n"
        "`/status` - System health & version\n"
        "`/dryrun` - Simulation run (no alerts)\n"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not check_auth(update): return
    
    # Read Fingerprint
    fingerprint_path = Path("audit_pack/RUN_LOGS/PROD_FINGERPRINT.log")
    fingerprint = "Unknown"
    if fingerprint_path.exists():
        with open(fingerprint_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "Git Commit" in line:
                    fingerprint = line.strip()
                    break
    
    # Read Scheduler Log (last line?)
    sched_log = Path("audit_pack/RUN_LOGS/scheduler.log")
    last_sched = "No logs"
    if sched_log.exists():
        # Get last line
        try:
             # simple tail
             with open(sched_log, 'rb') as f:
                 f.seek(-100, 2) # Go to end
                 last_sched = f.readlines()[-1].decode().strip()
        except:
             last_sched = "Read error"

    msg = (
        "🟢 *System Status*\n"
        f"Build: `{fingerprint}`\n"
        f"Scheduler: `{last_sched}`\n"
        f"Pipeline Lock: {'🔒 BUSY' if PIPELINE_LOCK else '✅ FREE'}\n"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

async def top_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not check_auth(update): return
    
    args = context.args
    top_n = 5
    if args and args[0].isdigit():
        top_n = int(args[0])
        
    csv_path = Path("audit_pack/A9_live/top_ev_bets.csv")
    if not csv_path.exists():
        await update.message.reply_text("⚠️ No bets file found (never run?).")
        return
        
    try:
        bets = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                bets.append(row)
        
        if not bets:
            await update.message.reply_text("💤 No bets in last run.")
            return
            
        # Format Table
        lines = ["Match                Pick           Odds  EV   Stake"]
        lines.append("-" * 60)
        
        for b in bets[:top_n]:
            match = f"{b.get('home_team')} vs {b.get('away_team')}"
            match = (match[:18] + '…') if len(match) > 18 else match.ljust(20)
            
            sel = b.get('selection', '')
            sel = (sel[:13] + '…') if len(sel) > 13 else sel.ljust(15)
            
            odds = f"{float(b.get('odds', 0)):.2f}".ljust(6)
            ev = f"{float(b.get('ev_pct', 0)):.0f}%".ljust(5)
            stake = f"{float(b.get('stake_pct', 0)):.0f}%" # Note: csv might have stake_pct as float string
            
            lines.append(f"{match} {sel} {odds} {ev} {stake}")
            
        table_str = "\n".join(lines)
        await update.message.reply_text(f"```\n{table_str}\n```", parse_mode=ParseMode.MARKDOWN)
        
    except Exception as e:
        logger.error(f"Error reading top bets: {e}")
        await update.message.reply_text("❌ Error reading bets file.")

async def run_pipeline_wrapper(update: Update, dry_run: bool = False):
    global PIPELINE_LOCK, LAST_RUN_TIME
    
    if PIPELINE_LOCK:
        await update.message.reply_text("⚠️ System is BUSY. Please wait.")
        return

    # Check cooldown (only for real runs)
    if not dry_run and LAST_RUN_TIME:
        elapsed = (datetime.utcnow() - LAST_RUN_TIME).total_seconds()
        if elapsed < RUN_COOLDOWN_SECONDS:
             mins = int((RUN_COOLDOWN_SECONDS - elapsed) / 60)
             await update.message.reply_text(f"⏳ Cooldown active. Wait {mins}m.")
             return

    # Lock
    PIPELINE_LOCK = True
    mode = "DRY RUN" if dry_run else "LIVE RUN"
    await update.message.reply_text(f"🚀 Starting {mode}...")
    
    try:
        cmd = ["python3", "scripts/run_value_finder.py", "--now", "--global-mode"]
        if dry_run:
            cmd.append("--dry-run")
        else:
            # For live run via bot, we generally want alerts to go to channel too?
            # User said "/now returns top bets to chat".
            # If we add --telegram, it sends to configured channel.
            # Let's add --telegram so it behaves like cron.
            cmd.append("--telegram")
            
        # Run subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300 # 5m timeout
        )
        
        output_snippet = result.stdout[-1000:] # Last 1000 chars
        
        if result.returncode == 0:
            await update.message.reply_text(f"✅ Run Completed.\n\nOutput snippet:\n```\n{output_snippet}\n```", parse_mode=ParseMode.MARKDOWN)
            if not dry_run:
                LAST_RUN_TIME = datetime.utcnow()
        else:
            await update.message.reply_text(f"❌ Run Failed (Code {result.returncode}).\nStderr:\n`{result.stderr[-500:]}`", parse_mode=ParseMode.MARKDOWN)
            
    except subprocess.TimeoutExpired:
        await update.message.reply_text("❌ Run Timed Out (5m limit).")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")
    finally:
        PIPELINE_LOCK = False

async def now_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not check_auth(update): return
    await run_pipeline_wrapper(update, dry_run=False)

async def dryrun_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not check_auth(update): return
    await run_pipeline_wrapper(update, dry_run=True)

def main():
    # Load Env
    load_env_config()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not found in .env")
        sys.exit(1)
        
    app = ApplicationBuilder().token(token).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("top", top_command))
    app.add_handler(CommandHandler("now", now_command))
    app.add_handler(CommandHandler("dryrun", dryrun_command))
    
    print("🤖 Bot is polling...")
    app.run_polling()

if __name__ == "__main__":
    main()
