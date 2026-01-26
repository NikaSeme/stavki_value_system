# RUNBOOK: Deploying Stavki V5

## 1. Requirements
- Python 3.9+ (`python3 --version`)
- Libraries: `pandas`, `catboost`, `requests`, `schedule` (see `requirements.txt`)
- Git

## 2. Installation
```bash
git clone <repo_url> stavki_value_system
cd stavki_value_system
pip install -r requirements.txt
```

## 3. Configuration
Create `.env`:
```ini
ODDS_API_KEY=your_key
TELEGRAM_BOT_TOKEN=123:ABC
TELEGRAM_CHAT_ID=-100123
```

## 4. Verification
Run a simulation (safe):
```bash
python3 scripts/run_value_finder.py --dry-run
```
output should show "TOP 5 BETS" table.

## 5. Enable Scheduling (Systemd)
Copy service files:
```bash
sudo cp deploy/systemd/stavki-v5.service /etc/systemd/system/
sudo cp deploy/systemd/stavki-v5.timer /etc/systemd/system/
sudo systemctl enable --now stavki-v5.timer
```

## 6. Maintenance
- **Logs**: `tail -f audit_pack/RUN_LOGS/scheduler.log`
- **Exposure**: `python3 scripts/calculate_roi.py`
- **Stop**: `sudo systemctl stop stavki-v5.timer`
