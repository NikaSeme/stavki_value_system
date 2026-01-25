#!/bin/bash
# STAVKI Strict Audit v3 Orchestrator
# Usage: ./scripts/run_audit.sh

set -e

# Configuration
LOG_DIR="audit_pack/RUN_LOGS"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo "🚀 STAVKI AUDIT V3 STARTED"
echo "=================================================="
echo "Timestamp: $(date)"
echo "Log Dir: $LOG_DIR"
echo "=================================================="

# Helper
run_step() {
    NAME=$1
    LOG_FILE="$LOG_DIR/$2"
    CMD=$3
    
    echo "▶ Running $NAME..."
    echo "  Command: $CMD"
    
    # Run, capture output/error to log, also display error if fails
    if eval "$CMD" > "$LOG_FILE" 2>&1; then
        echo "  ✅ PASSED"
    else
        echo "  ❌ FAILED (See $LOG_FILE)"
        tail -n 10 "$LOG_FILE"
        exit 1
    fi
}

# 1. Installation & Environment
run_step "A1: Env Capture" "00_env.log" "pip freeze > audit_pack/A0_identity/env_info.txt && python3 --version >> audit_pack/A0_identity/env_info.txt"

# 2. Unit Tests
run_step "Tests: Unit" "02_tests.log" "pytest tests/"

# 3. Model Load Smoke Test
run_step "A9: Model Smoke" "model_load.log" "python3 scripts/diagnostics/model_load_smoke.py"

# 3.5 Live Pipeline Dry Run (Populate artifacts)
run_step "A4: Live Odds Fetch" "live_odds.log" "python3 scripts/run_odds_pipeline.py --sport soccer_epl --hours-ahead 48"
run_step "A9: Live Value Dry Run" "live_value.log" "python3 scripts/run_value_finder.py --sport soccer_epl --ev-threshold 0.01 --top-n 5"
cp audit_pack/RUN_LOGS/live_value.log audit_pack/A9_live/run_scheduler_log.txt

# 4. Odds Integrity
run_step "A4: Odds Integrity" "odds_integrity.log" "python3 checks/run_odds_integrity.py"

# 5. Time Integrity & Sanity
run_step "A3: Time Check" "time_integrity.log" "python3 checks/01_feature_time_check.py"
run_step "A3: Sanity Check" "prob_sanity.log" "python3 checks/run_sanity_check.py"

# 6. Metrics & Models
run_step "A6: Metrics" "metrics.log" "python3 checks/run_metrics_evaluation.py"
run_step "A5: Models Discovery" "models.log" "python3 checks/run_models_discovery.py"

# 7. Backtest
run_step "A7: Backtest" "backtest.log" "python3 checks/run_backtest_audit.py"

# 8. Staking
run_step "A8: Staking" "staking.log" "python3 checks/run_staking_audit.py"

# 9. Pack Validation
run_step "A10: Pack Validation" "validate_pack.log" "python3 scripts/validate_audit_pack.py"

# 10. Zip
echo "▶ Zipping Audit Pack..."
rm -f stavki_value_cleaned_pack_v3.zip
zip -r stavki_value_cleaned_pack_v3.zip audit_pack/ reports/ src/ scripts/ config/
echo "✅ Zip created: stavki_value_cleaned_pack_v3.zip"

echo "=================================================="
echo "🎉 AUDIT V3 SUCCESS"
echo "=================================================="
