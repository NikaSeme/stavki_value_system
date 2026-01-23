#!/bin/bash
# Setup script for Google Compute Engine (Debian/Ubuntu)
# Run this script on the server after cloning the repo

set -e  # Exit on error

echo "=================================================="
echo "🚀 STAVKI System Setup (GCE)"
echo "=================================================="

# 1. Update System
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# 2. Install Dependencies
echo "📦 Installing Python & Git..."
sudo apt-get install -y python3 python3-venv python3-pip git htop

# 3. Setup Virtual Environment
echo "🐍 Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   ✓ Virtual environment created"
else
    echo "   ✓ Virtual environment already exists"
fi

# 4. Install Project Requirements
echo "📥 Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found! Installing basics..."
    pip install pandas numpy requests python-dotenv python-telegram-bot
fi

# 5. Create Directories
echo "📂 Creating output directories..."
mkdir -p outputs/logs outputs/data outputs/models outputs/state

# 6. Check .env
if [ ! -f ".env" ]; then
    echo "⚠️  .env file missing!"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "   ✓ Created .env from example (PLEASE EDIT IT)"
    else
        echo "   ❌ No .env.example found. You must create .env manually."
    fi
else
    echo "✓ .env file exists"
fi

echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo "Next steps:"
echo "1. Edit .env with your real API keys: nano .env"
echo "2. Install cron jobs: crontab -e"
echo "3. Run scheduler: source venv/bin/activate && python run_scheduler.py --telegram"
