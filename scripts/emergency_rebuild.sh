
#!/bin/bash
set -e  # Exit immediately if any command fails

echo "========================================================"
echo "🧹 STAVKI EMERGENCY REBUILD & REPAIR (v1.0)"
echo "========================================================"

echo ""
echo "STEP 0: CLEANING GIT STATE"
echo "--------------------------"
# Force stash any local changes that might be blocking pull
git stash || echo "Warning: Git stash failed (maybe nothing to stash), continuing..."
git pull origin main || { 
    echo "❌ Git Pull Failed. You might have untracked file conflicts."
    echo "   Attempting to resolve by forcing overwrite of scripts..."
    git checkout origin/main -- scripts/bridge_features.py
    echo "   ✓ Forced checkout of bridge_features.py"
}

echo ""
echo "STEP 1: BRIDGING FEATURES (Legacy Format)"
echo "----------------------------------------"
python3 scripts/bridge_features.py

echo ""
echo "STEP 2: TRAINING MODEL B (CatBoost)"
echo "-----------------------------------"
python3 scripts/train_model.py

# Verification Step
if [ ! -f "models/catboost_v1_latest.pkl" ]; then
    echo "❌ CRITICAL FAILURE: models/catboost_v1_latest.pkl was NOT created."
    echo "   Check the logs above for 'KeyError' or other crashes."
    exit 1
fi
echo "✓ Verified: CatBoost model exists."

echo ""
echo "STEP 3: TRAINING ENSEMBLE (Model C)"
echo "-----------------------------------"
python3 scripts/train_simple_ensemble.py

# Verification Step
if [ ! -f "models/ensemble_simple_latest.pkl" ]; then
    echo "❌ CRITICAL FAILURE: models/ensemble_simple_latest.pkl was NOT created."
    exit 1
fi
echo "✓ Verified: Ensemble model exists."

echo ""
echo "========================================================"
echo "✅ ALL SYSTEMS GO! LAUNCHING BOT..."
echo "========================================================"
python3 scripts/run_value_finder.py --now --global-mode --auto
