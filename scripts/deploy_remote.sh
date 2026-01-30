#!/bin/bash
# STAVKI Production Deployment Script
# Usage: ./scripts/deploy_remote.sh [USER@HOST] [SSH_KEY_OPTIONAL]

TARGET=$1
SSH_KEY=$2

if [ -z "$TARGET" ]; then
    echo "❌ Usage: ./scripts/deploy_remote.sh [USER@HOST] [path/to/key]"
    echo "Example: ./scripts/deploy_remote.sh ubuntu@123.45.67.89"
    exit 1
fi

SSH_OPTS="-o StrictHostKeyChecking=no"
if [ ! -z "$SSH_KEY" ]; then
    SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
fi

REMOTE_DIR="~/stavki_value_system"

echo "========================================"
echo "🚀 Deploying to $TARGET"
echo "========================================"

# 1. Sync Files
echo "📦 Syncing files..."
rsync -avz -e "ssh $SSH_OPTS" \
    --exclude 'venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'audit_pack/RUN_LOGS' \
    --exclude 'data/odds/*.db' \
    --exclude 'models/catboost_*.pkl' \
    ./ $TARGET:$REMOTE_DIR

# Note: We exclude massive models/DBs locally if they are meant to be generated on server.
# BUT, user probably wants current state. Removing exclusion for DB/models if desirable.
# Correcting: Let's sync models, but maybe keep DB local-only or sync it if it's the "Golden Source".
# For now, excluding logs/tmp is safe.

echo "✅ Files synced."

# 2. Remote Setup
echo "🔧 Configuring Remote Server..."
ssh $SSH_OPTS $TARGET << EOF
    # Update & Install Deps
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-venv

    cd $REMOTE_DIR

    # Virtual Env
    if [ ! -d "venv" ]; then
        echo "Creating venv..."
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    
    # Install Python Deps
    echo "Installing requirements..."
    pip install -r requirements.txt
    
    # Install Systemd Services
    echo "Installing Systemd services..."
    
    # Fix paths in service files if needed (assuming /home/ubuntu, replace with \$HOME)
    # Actually, service files have hardcoded paths. We should sed them.
    CURRENT_USER=\$(whoami)
    CURRENT_HOME=\$HOME
    
    # Adjust Stavki Bot Service
    sed -i "s|User=ubuntu|User=\$CURRENT_USER|g" deploy/stavki-bot.service
    sed -i "s|Group=ubuntu|Group=\$CURRENT_USER|g" deploy/stavki-bot.service
    sed -i "s|/home/ubuntu/stavki_value_system|\$CURRENT_HOME/stavki_value_system|g" deploy/stavki-bot.service
    
    # Adjust Scheduler Service
    sed -i "s|User=ubuntu|User=\$CURRENT_USER|g" deploy/stavki-scheduler.service
    sed -i "s|Group=ubuntu|Group=\$CURRENT_USER|g" deploy/stavki-scheduler.service
    sed -i "s|/home/ubuntu/stavki_value_system|\$CURRENT_HOME/stavki_value_system|g" deploy/stavki-scheduler.service
    
    # Copy to /etc/systemd/system
    sudo cp deploy/stavki-bot.service /etc/systemd/system/
    sudo cp deploy/stavki-scheduler.service /etc/systemd/system/
    
    # Reload & Restart
    sudo systemctl daemon-reload
    sudo systemctl enable stavki-bot stavki-scheduler
    sudo systemctl restart stavki-bot stavki-scheduler
    
    echo "✅ Remote Setup Complete!"
    
    # Status Check
    systemctl status stavki-bot --no-pager
EOF

echo "========================================"
echo "🎉 Deployment Finished Successfully"
echo "========================================"
