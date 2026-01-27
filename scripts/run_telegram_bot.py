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
import json
from datetime import datetime
from dotenv import load_dotenv

import yaml
try:
    from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
    from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
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
        "/stop - Emergency stop active pipeline\n"
        "/choose_leagues - Trigger run for specific leagues\n"
        "/set_bankroll <eur> - Set persistent budget\n"
        "/set_ev <0.xx> - Set persistent EV threshold\n"
        "/status - Check system status & settings\n"
        "/help - Show all commands"
    )
    await update.message.reply_text(msg, parse_mode='Markdown')

def load_user_settings():
    path = Path("config/user_settings.json")
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"bankroll": 40.0, "ev_threshold": 0.08}

def save_user_settings(settings):
    path = Path("config/user_settings.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(settings, f, indent=2)

async def set_bankroll(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set persistent bankroll."""
    if not check_auth(update.effective_user.id): return
    if not context.args:
        await update.message.reply_text("Usage: `/set_bankroll 50`", parse_mode='Markdown')
        return
    try:
        val = float(context.args[0])
        settings = load_user_settings()
        settings['bankroll'] = val
        save_user_settings(settings)
        await update.message.reply_text(f"✅ Bankroll updated to **{val}€**", parse_mode='Markdown')
    except ValueError:
        await update.message.reply_text("❌ Please enter a valid number.")

async def set_ev(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set persistent EV threshold."""
    if not check_auth(update.effective_user.id): return
    if not context.args:
        await update.message.reply_text("Usage: `/set_ev 0.10` (for 10%)", parse_mode='Markdown')
        return
    try:
        val = float(context.args[0])
        settings = load_user_settings()
        settings['ev_threshold'] = val
        save_user_settings(settings)
        await update.message.reply_text(f"✅ EV Threshold updated to **{int(val*100)}%**", parse_mode='Markdown')
    except ValueError:
        await update.message.reply_text("❌ Please enter a valid number.")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help message."""
    if not check_auth(update.effective_user.id): return
    msg = (
        "📖 *Help*\n\n"
        "`/run` - Runs odds fetch and value finder.\n"
        "`/run <bankroll> <ev>` - Run with temporary overrides.\n"
        "`/choose_leagues` - Interactive menu to select specific competitions.\n"
        "`/stop` - Force stops active pipeline and clears locks.\n"
        "`/set_bankroll <eur>` - Updates your saved budget.\n"
        "`/set_ev <0.xx>` - Updates your saved EV threshold.\n"
        "`/status` - Basic system health check."
    )
    await update.message.reply_text(msg, parse_mode='Markdown')

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """System status."""
    if not check_auth(update.effective_user.id): return
    
    # Check for model
    model_exists = Path("models/catboost_v1_latest.pkl").exists()
    lock_exists = Path("/tmp/stavki_scheduler.lock").exists()
    settings = load_user_settings()
    
    msg = (
        "🔍 *Status*\n\n"
        f"Model: {'🟢 Ready' if model_exists else '🔴 Missing'}\n"
        f"Scheduler: {'🟡 Running' if lock_exists else '⚪ Idle'}\n"
        f"Bankroll: `{settings['bankroll']}€` (Saved)\n"
        f"EV Threshold: `{int(settings['ev_threshold']*100)}%` (Saved)\n"
        f"Time: {datetime.utcnow().strftime('%H:%M UTC')}"
    )
    await update.message.reply_text(msg, parse_mode='Markdown')

def get_available_leagues():
    """Load leagues from config/leagues.yaml"""
    path = Path("config/leagues.yaml")
    if not path.exists(): return []
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        leagues = []
        # Support only soccer for now as per policy (M66)
        for sport in ['soccer']:
            if sport in data:
                for league in data[sport]:
                    if league.get('active', True):
                        name = league['name']
                        # Shorten long names (M67)
                        name = name.replace("UEFA ", "").replace("English ", "")
                        leagues.append({'key': league['key'], 'name': name})
        return leagues
    except Exception as e:
        logger.error(f"Error loading leagues: {e}")
        return []

async def choose_leagues(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show interactive league selection menu."""
    if not check_auth(update.effective_user.id): return
    
    leagues = get_available_leagues()
    if not leagues:
        await update.message.reply_text("❌ No active leagues found in config.")
        return
        
    # Initialize selection if missing
    if 'selected_leagues' not in context.user_data:
        context.user_data['selected_leagues'] = set()
    
    await send_league_menu(update, context)

async def send_league_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Helper to render/update the selection menu."""
    leagues = get_available_leagues()
    selected = context.user_data.get('selected_leagues', set())
    
    keyboard = []
    # 1 column for better fit on all devices (M67)
    for l in leagues:
        text = f"{'✅ ' if l['key'] in selected else '⬜ '}{l['name']}"
        keyboard.append([InlineKeyboardButton(text, callback_data=f"toggle_{l['key']}")])
    
    # Selection helpers
    keyboard.append([
        InlineKeyboardButton("✨ SELECT ALL", callback_data="all_select"),
        InlineKeyboardButton("🧹 CLEAR ALL", callback_data="all_clear")
    ])
        
    keyboard.append([InlineKeyboardButton("🚀 RUN SELECTED", callback_data="run_targeted")])
    keyboard.append([InlineKeyboardButton("❌ CANCEL", callback_data="cancel_selection")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    text = "🎯 *Select Protocols for Custom Run*\n(Toggle and then click Run)"
    
    try:
        if update.callback_query:
            await update.callback_query.edit_message_text(
                text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text(
                text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
    except Exception as e:
        logger.error(f"Menu update failed: {e}")

async def league_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle menu interactions."""
    query = update.callback_query
    if not query: return
    
    try:
        await query.answer()
        data = query.data
        selected = context.user_data.get('selected_leagues', set())
        
        if data.startswith("toggle_"):
            league_key = data.replace("toggle_", "")
            if league_key in selected:
                selected.remove(league_key)
            else:
                selected.add(league_key)
            context.user_data['selected_leagues'] = selected
            await send_league_menu(update, context)
            
        elif data == "all_select":
            available = get_available_leagues()
            context.user_data['selected_leagues'] = {l['key'] for l in available}
            await send_league_menu(update, context)
            
        elif data == "all_clear":
            context.user_data['selected_leagues'] = set()
            await send_league_menu(update, context)
        
        elif data == "run_targeted":
            if not selected:
                await query.edit_message_text("❌ Please select at least one league.")
                return
                
            leagues_str = ",".join(selected)
            settings = load_user_settings()
            bankroll = settings.get('bankroll', 40.0)
            ev = settings.get('ev_threshold', 0.08)
            
            await query.edit_message_text(f"🔄 *Triggering Targeted Run...*\nSelected: `{len(selected)}` protocols", parse_mode='Markdown')
            
            cmd = [
                sys.executable, "scripts/run_scheduler.py", 
                "--now", 
                "--telegram",
                "--bankroll", str(bankroll),
                "--ev-threshold", str(ev),
                "--leagues", leagues_str
            ]
            
            subprocess.Popen(cmd)
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="✅ *Targeted Run triggered!* Check progress below.",
                parse_mode='Markdown'
            )
            context.user_data['selected_leagues'] = set()

        elif data == "cancel_selection":
            context.user_data['selected_leagues'] = set()
            await query.edit_message_text("🚫 Selection cancelled.")
            
    except Exception as e:
        logger.error(f"Callback error: {e}")
        try:
            await context.bot.send_message(chat_id=query.message.chat_id, text=f"⚠️ *Menu Error:* {str(e)}")
        except: pass

async def run_pipeline(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Trigger the pipeline."""
    if not check_auth(update.effective_user.id): return
    
    settings = load_user_settings()
    bankroll = settings.get('bankroll', 40.0)
    ev = settings.get('ev_threshold', 0.08)
    
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

async def stop_pipeline(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Force stop any active pipeline processes."""
    if not check_auth(update.effective_user.id): return
    
    await update.message.reply_text("🛑 *Attempting to stop pipeline...*")
    
    # 1. Kill processes
    scripts_to_kill = ["run_scheduler.py", "run_value_finder.py", "run_odds_pipeline.py"]
    killed_any = False
    
    try:
        import signal
        # Use pkill -f to find scripts by name in command line
        for script in scripts_to_kill:
            # We use subprocess with pkill
            subprocess.run(["pkill", "-f", script])
            killed_any = True
        
        # 2. Clear lock file
        LOCK_FILE = Path("/tmp/stavki_scheduler.lock")
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
            await update.message.reply_text("🔓 Lock file cleared.")
        
        await update.message.reply_text("✅ *System Stopped.* You can now start a fresh run.")
        logger.info(f"User {update.effective_user.id} triggered emergency stop.")
        
    except Exception as e:
        await update.message.reply_text(f"⚠️ *Cleanup Error:* {str(e)}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true", help="Check system status and exit")
    args = parser.parse_args()

    # Load environment first for check
    load_dotenv(ENV_FILE, override=True)

    if args.check_only:
        model_exists = Path("models/catboost_v1_latest.pkl").exists()
        if model_exists:
            print("✅ Model: Ready")
            sys.exit(0)
        else:
            print("❌ Model: Missing")
            sys.exit(1)

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
    app.add_handler(CommandHandler("stop", stop_pipeline))
    app.add_handler(CommandHandler("choose_leagues", choose_leagues))
    app.add_handler(CommandHandler("set_bankroll", set_bankroll))
    app.add_handler(CommandHandler("set_ev", set_ev))
    app.add_handler(CallbackQueryHandler(league_callback))
    
    app.run_polling()

if __name__ == "__main__":
    main()
