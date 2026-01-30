#!/bin/bash
# STAVKI Server Setup (Run this ON the server)
# Usage: sudo ./scripts/setup_server.sh

echo "🔧 Configuring STAVKI Server..."

# 1. Verify we are in the right dir
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Run this from the project root (where requirements.txt is)."
    exit 1
fi

# 2. Install Dependencies (System)
echo "📦 Installing system dependencies..."
apt-get update
apt-get install -y python3-pip python3-venv

# 3. Setup Virtualenv if missing
if [ ! -d "venv" ]; then
    echo "🐍 Creating venv..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "✅ Venv exists. Updating requirements..."
    source venv/bin/activate
    pip install -r requirements.txt
fi

# 3.1 Generate Initial Models (if missing)
echo "🧠 Checking Models..."
source venv/bin/activate
if [ ! -f "models/catboost_v1_latest.pkl" ]; then
    echo "   -> Generating initial models (this may take a minute)..."
    # Ensure src is in path
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    python3 scripts/train_model.py
    echo "✅ Models generated."
else
    echo "   -> Models exist. Skipping generation."
fi

# 4. Configure Systemd
echo "⚙️ Configuring Systemd Services..."

# Determine User and Path
CURRENT_USER=${SUDO_USER:-$(whoami)}
PROJECT_DIR=$(pwd)

echo "   -> User: $CURRENT_USER"
echo "   -> Path: $PROJECT_DIR"

# Modify Service Files dynamically
sed -i "s|User=ubuntu|User=$CURRENT_USER|g" deploy/stavki-bot.service
sed -i "s|Group=ubuntu|Group=$CURRENT_USER|g" deploy/stavki-bot.service
sed -i "s|/home/ubuntu/stavki_value_system|$PROJECT_DIR|g" deploy/stavki-bot.service

sed -i "s|User=ubuntu|User=$CURRENT_USER|g" deploy/stavki-scheduler.service
sed -i "s|Group=ubuntu|Group=$CURRENT_USER|g" deploy/stavki-scheduler.service
sed -i "s|/home/ubuntu/stavki_value_system|$PROJECT_DIR|g" deploy/stavki-scheduler.service

# Fix Python Path (Use venv instead of system python)
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
sed -i "s|/usr/bin/python3|$VENV_PYTHON|g" deploy/stavki-bot.service
sed -i "s|/usr/bin/python3|$VENV_PYTHON|g" deploy/stavki-scheduler.service

# Copy to Systemd
cp deploy/stavki-bot.service /etc/systemd/system/
cp deploy/stavki-scheduler.service /etc/systemd/system/

# 5. Enable and Start
echo "🚀 Starting Services..."
systemctl daemon-reload
systemctl enable stavki-bot stavki-scheduler
systemctl restart stavki-bot stavki-scheduler

echo "=================================="
echo "✅ Server Setup Complete!"
echo "Status Check:"
systemctl status stavki-bot --no-pager
echo "=================================="
