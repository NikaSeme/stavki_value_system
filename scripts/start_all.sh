#!/bin/bash
# STAVKI System Startup Script
# Usage: ./scripts/start_all.sh

# Ensure logs dir exists
mkdir -p audit_pack/RUN_LOGS

echo "=================================="
echo "🚀 STAVKI System Startup"
echo "=================================="

# 1. Check & Kill existing (to avoid partial state)
echo "🧹 Cleaning up old processes..."
pkill -f run_scheduler.py
pkill -f run_telegram_bot.py
# Clear locks
rm -f /tmp/stavki_scheduler.lock

# 2. Start Scheduler
echo "⏰ Starting Scheduler (Interval: 60m + 10m checks)..."
nohup python3 scripts/run_scheduler.py --interval 60 --telegram >> audit_pack/RUN_LOGS/scheduler.log 2>&1 &
SCHED_PID=$!
echo "   -> PID: $SCHED_PID"

# 3. Start Bot Interface
echo "🤖 Starting Telegram Bot Interface..."
nohup python3 scripts/run_telegram_bot.py >> audit_pack/RUN_LOGS/telegram_bot.log 2>&1 &
BOT_PID=$!
echo "   -> PID: $BOT_PID"

echo "=================================="
echo "✅ System Live!"
echo "   Type: tail -f audit_pack/RUN_LOGS/scheduler.log"
echo "=================================="
