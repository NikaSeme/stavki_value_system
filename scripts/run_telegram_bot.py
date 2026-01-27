#!/usr/bin/env python3
"""
Interactive Telegram Bot for STAVKI.
Allows users to trigger runs and check status via Telegram.
"""
import os
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes
except ImportError:
    print("Error: python-telegram-bot not installed. Run 'pip install python-telegram-bot'")
    sys.exit(1)

# Load environment
load_dotenv()
ENV_FILE = "/etc/stavki/stavki.env"
if os.path.exists(ENV_FILE):
    load_dotenv(ENV_FILE, override=True)

# Logging
log_dir = Path("audit_pack/RUN_LOGS")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=log_dir / "telegram_bot.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StavkiBot")

# Auth
ALLOWED_USER_ID = os.getenv('TELEGRAM_CHAT_ID')
if ALLOWED_USER_ID:
    try:
        ALLOWED_USER_IDS = [int(ALLOWED_USER_ID)]
    except ValueError:
        ALLOWED_USER_IDS = []
else:
    ALLOWED_USER_IDS = []

def check_auth(user_id):
    return user_id in ALLOWED_USER_IDS

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome message."""
    user_id = update.effective_user.id
    if not check_auth(user_id):
        await update.message.reply_text("⛔ Unauthorized.")
        return
    
    msg = (
        "🎯 *STAVKI Betting Bot*\n\n"
        "Commands:\n"
        "/run - Start a fresh pipeline run\n"
        "/status - Check system status\n"
        "/help - Show all commands"
    )
    await update.message.reply_text(msg, parse_mode='Markdown')

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help message."""
    if not check_auth(update.effective_user.id): return
    msg = (
        "📖 *Help*\n\n"
        "`/run` - Runs odds fetch and value finder.\n"
        "`/run <bankroll> <ev>` - Run with overrides (e.g. `/run 50 0.10`)\n"
        "`/status` - Basic system health check."
    )
    await update.message.reply_text(msg, parse_mode='Markdown')

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """System status."""
    if not check_auth(update.effective_user.id): return
    
    # Check for model
    model_exists = Path("models/catboost_v1_latest.pkl").exists()
    lock_exists = Path("/tmp/stavki_scheduler.lock").exists()
    
    msg = (
        "🔍 *Status*\n\n"
        f"Model: {'🟢 Ready' if model_exists else '🔴 Missing'}\n"
        f"Scheduler: {'🟡 Running' if lock_exists else '⚪ Idle'}\n"
        f"Time: {datetime.utcnow().strftime('%H:%M UTC')}"
    )
    await update.message.reply_text(msg, parse_mode='Markdown')

async def run_pipeline(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Trigger the pipeline."""
    if not check_auth(update.effective_user.id): return
    
    bankroll = 40.0
    ev = 0.08
    
    if context.args:
        try:
            bankroll = float(context.args[0])
            if len(context.args) > 1:
                ev = float(context.args[1])
        except ValueError:
            await update.message.reply_text("❌ Invalid format. Use: `/run 40 0.08`")
            return

    await update.message.reply_text(f"🔄 *Starting Run...*\nBudget: {bankroll}€ | Min EV: {int(ev*100)}%")
    
    # We trigger the scheduler in '--now' mode so it handles both steps
    cmd = [
        sys.executable, "scripts/run_scheduler.py", 
        "--now", 
        "--telegram",
        "--bankroll", str(bankroll),
        "--ev-threshold", str(ev)
    ]
    
    try:
        # Run in background to not block the bot
        subprocess.Popen(cmd)
        await update.message.reply_text("✅ *Run triggered!* You will receive results in this chat shortly.")
    except Exception as e:
        await update.message.reply_text(f"💥 *Error:* {str(e)}")

def main():
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN not found in environment.")
        sys.exit(1)
        
    print(f"🤖 Bot starting... (Allowed User: {ALLOWED_USER_ID})")
    app = Application.builder().token(token).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("run", run_pipeline))
    
    app.run_polling()

if __name__ == "__main__":
    main()
