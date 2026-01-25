#!/bin/bash
# STAVKI Strict Forensic Audit v2 Orchestrator
# Usage: ./scripts/run_audit_v2.sh --mode audit

set -e

MODE=$1
LOG_DIR="outputs/audit_v2/RUN_LOGS"
AUDIT_DIR="outputs/audit_v2"

mkdir -p "$LOG_DIR"
mkdir -p "$AUDIT_DIR/checks"

echo "=================================================="
echo "🚀 STAVKI AUDIT V2 STARTED"
echo "=================================================="
echo "Mode: $MODE"
echo "Log Dir: $LOG_DIR"
echo "Timestamp: $(date)"
echo "=================================================="

# Function to run command and log
run_step() {
    STEP_NAME=$1
    LOG_FILE="$LOG_DIR/$2"
    CMD=$3

    echo "▶ Running $STEP_NAME..."
    echo "  Command: $CMD"
    echo "  Log: $LOG_FILE"
    
    # Start log
    echo "=== START: $STEP_NAME at $(date) ===" > "$LOG_FILE"
    
    # Run command, capture stdout/stderr to log, and also print to screen if failed
    if eval "$CMD" >> "$LOG_FILE" 2>&1; then
        echo "  ✅ PASSED"
        echo "=== END: $STEP_NAME at $(date) ===" >> "$LOG_FILE"
    else
        echo "  ❌ FAILED (See $LOG_FILE)"
        echo "=== FAILED: $STEP_NAME at $(date) ===" >> "$LOG_FILE"
        # Optional: Don't exit on failure if we want to gather as much evidence as possible
        # exit 1 
    fi
}

if [ "$MODE" == "--mode" ] && [ "$2" == "audit" ]; then
    
    # Phase 0: Identity (Shell commands)
    run_step "Identity: Repo Tree" "00_repo_tree.log" "tree -a -L 6 -I 'venv|__pycache__|.git' > $AUDIT_DIR/repo_tree.txt"
    run_step "Identity: Git Status" "00_git_status.log" "git status > $AUDIT_DIR/git_status.txt && git log -n 10 > $AUDIT_DIR/git_log_10.txt"
    run_step "Identity: Env Info" "00_env_info.log" "python3 --version > $AUDIT_DIR/env_info.txt && pip freeze >> $AUDIT_DIR/env_info.txt && uname -a >> $AUDIT_DIR/env_info.txt"

    # Phase 1: Installation & Tests
    run_step "A1: Install" "01_install.log" "pip install -r requirements.txt"
    run_step "A1: Tests" "02_tests.log" "pytest tests/ -v"

    # Phase 3: Leakage Checks (Python)
    # Using '|| true' to ensure script continues even if checks fail (we want to record the failure)
    run_step "A3: Schema Check" "10_schema_check.log" "python3 checks/00_schema_check.py || true"
    run_step "A3: Feature Time Check" "11_feature_time_check.log" "python3 checks/01_feature_time_check.py || true"
    run_step "A3: Rolling Leakage" "12_rolling_leakage_check.log" "python3 checks/02_rolling_leakage_check.py || true"

    # Phase 4: Odds Integrity
    run_step "A4: Odds Integrity" "20_odds_integrity.log" "python3 checks/run_odds_integrity.py || true"

    # Phase 5: Models Discovery
    run_step "A5: Models Discovery" "30_models_discovery.log" "python3 checks/run_models_discovery.py || true"

    # Phase 6: Metrics
    run_step "A6: Metrics" "40_metrics.log" "python3 checks/run_metrics_evaluation.py || true"

    # Phase 7: Backtest
    run_step "A7: Backtest" "50_backtest.log" "python3 checks/run_backtest_audit.py || true"
    
    # Phase 8: Staking
    run_step "A8: Staking Checks" "60_staking.log" "python3 checks/run_staking_audit.py || true"

    echo "=================================================="
    echo "✅ AUDIT EXECUTION COMPLETE"
    echo "Check $LOG_DIR for details."
    echo "=================================================="

else
    echo "Usage: ./scripts/run_audit_v2.sh --mode audit"
    exit 1
fi
