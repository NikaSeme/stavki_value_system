#!/bin/bash
# STAVKI Server Setup & Deployment (M33)
# Automates directory creation and systemd unit generation.

# Exit on error
set -e

PROJECT_ROOT=$(pwd)
SERVICE_USER=$USER
LOG_DIR="/var/log/stavki"
LIB_DIR="/var/lib/stavki"
ENV_FILE="/etc/stavki/stavki.env"

echo "===================================================="
echo "STAVKI PROD SERVER SETUP"
echo "===================================================="

# 1. Directories
echo "1. Creating Directories..."
sudo mkdir -p $LOG_DIR $LIB_DIR /etc/stavki
sudo chown $SERVICE_USER:$SERVICE_USER $LOG_DIR $LIB_DIR /etc/stavki
mkdir -p audit_pack/RUN_LOGS outputs/odds logs

# 2. Environment Template
if [ ! -f "$ENV_FILE" ]; then
    echo "2. Generating Environment Template at $ENV_FILE..."
    sudo bash -c "cat <<EOF > $ENV_FILE
# STAVKI Production Environment
ODDS_API_KEY=your_key_here
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
STAVKI_AUTOMATED=1
PYTHONUNBUFFERED=1
EOF"
    sudo chmod 600 $ENV_FILE
    sudo chown $SERVICE_USER:$SERVICE_USER $ENV_FILE
    echo "⚠️  ACTION REQUIRED: Edit $ENV_FILE with your secrets."
else
    echo "2. $ENV_FILE already exists. Skipping."
fi

# 3. Systemd Service Template (Main Pipeline)
echo "3. Generating Systemd Services..."
sudo bash -c "cat <<EOF > /etc/systemd/system/stavki.service
[Unit]
Description=Stavki Value Betting Pipeline
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=$SERVICE_USER
WorkingDirectory=$PROJECT_ROOT
EnvironmentFile=$ENV_FILE
ExecStart=$PROJECT_ROOT/venv/bin/python3 scripts/run_scheduler.py --now --telegram --bankroll 40 --ev-threshold 0.08
ExecStart=$PROJECT_ROOT/venv/bin/python3 scripts/cleanup_maintenance.py --days 14
TimeoutStartSec=600

[Install]
WantedBy=multi-user.target
EOF"

# 4. Systemd Service Template (Interactive Bot)
sudo bash -c "cat <<EOF > /etc/systemd/system/stavki-bot.service
[Unit]
Description=Stavki Interactive Telegram Bot
After=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$PROJECT_ROOT
EnvironmentFile=$ENV_FILE
ExecStart=$PROJECT_ROOT/venv/bin/python3 scripts/run_telegram_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF"

# 5. Systemd Timer Template (Daily Runs)
echo "5. Generating Systemd Timer (Every 6 hours)..."
sudo bash -c "cat <<EOF > /etc/systemd/system/stavki.timer
[Unit]
Description=Run Stavki on schedule

[Timer]
OnCalendar=00,06,12,18:00:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
EOF"

echo "6. Reloading systemd..."
sudo systemctl daemon-reload

echo "===================================================="
echo "SETUP COMPLETE"
echo "===================================================="
echo "Next steps:"
echo "1. Verify secrets in $ENV_FILE"
echo "2. Enable timer: sudo systemctl enable --now stavki.timer"
echo "3. Check status: systemctl status stavki.timer"
echo "4. Test run: sudo systemctl start stavki.service"
echo "===================================================="
