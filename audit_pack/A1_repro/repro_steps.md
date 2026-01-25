# Reproducibility Steps (Audit v3)

## 1. Environment Setup
```bash
# Clone repository
git clone <repo_url>
cd stavki_value_system

# Setup Python 3.9+
python3 -m venv venv
source venv/bin/activate

# Install dependencies (pinned)
pip install -r requirements.txt
```

## 2. Full Audit Execution ("One Button")
We provide a master script that runs the entire verification chain (tests, diagnostics, backtest, validation).

```bash
./scripts/run_audit.sh
```

This will:
1.  Install dependencies
2.  Run unit tests (`pytest`)
3.  Execute forensic checks (Time, Odds, Staking)
4.  Generate metrics and plots
5.  Run backtest simulation
6.  Validate the output pack structure
7.  Zip the results to `stavki_value_cleaned_pack_v3.zip`

## 3. Manual Reproduction
If you wish to run individual steps:

### A. Run Backtest
```bash
python checks/run_backtest_audit.py
# Output: audit_pack/A7_backtest/bets_backtest.csv
```

### B. Live Pipeline Debug
```bash
python scripts/run_scheduler.py --dry-run
# Output: audit_pack/A9_live/predictions.csv (no alerts sent)
```

### C. Odds Pipeline
```bash
python scripts/run_odds_pipeline.py --sport soccer_epl
# Output: Data in data/raw and data/processed
```
