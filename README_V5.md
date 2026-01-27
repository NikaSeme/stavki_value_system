# Stavki V5 (Production Release)

**Version**: v5.0 (Vienna Schedule)
**Docs**: [AUDIT_FINAL_v5.md](AUDIT_FINAL_v5.md)

## 1. Quick Start

### Run Immediately (On-Demand)
```bash
python3 scripts/run_scheduler.py --now --telegram
```

### Dry Run (Simulation)
```bash
python3 scripts/run_value_finder.py --dry-run
```

### View Top Bets (No Alerts)
```bash
python3 scripts/run_value_finder.py --top 10
```

---

## 2. Production Deployment (Vienna Schedule)

The system is configured to run at **12:00** and **22:00** Europe/Vienna time.

### Option A: Systemd (Recommended)
1. Copy updated files to `/etc/systemd/system/`:
   ```bash
   sudo cp deploy/systemd/stavki-v5.service /etc/systemd/system/
   sudo cp deploy/systemd/stavki-v5.timer /etc/systemd/system/
   ```
2. Enable and start:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now stavki-v5.timer
   ```
3. Check status:
   ```bash
   systemctl list-timers --all
   ```

### Option B: Cron
1. Edit crontab:
   ```bash
   crontab -e
   ```
2. Add lines (ensure `CRON_TZ` is supported or adjust to UTC):
   ```bash
   CRON_TZ=Europe/Vienna
   0 12,22 * * * cd /opt/stavki_value_system && /usr/bin/python3 scripts/run_value_finder.py --now --telegram >> /var/log/stavki.log 2>&1
   ```

---

## 3. Configuration & Secrets
Ensure `.env` exists in root:
```ini
ODDS_API_KEY=your_key_here
TELEGRAM_BOT_TOKEN=123456:ABC-DEF
TELEGRAM_CHAT_ID=-100123456789
TELEGRAM_ADMIN_IDS=12345678,87654321  # Optional: Comma-separated user IDs for whitelist
```

## 4. Interactive Command Bot
Run the bot to allow on-demand commands (`/now`, `/status`, `/top`).
```bash
nohup python3 scripts/run_telegram_bot.py &
```
Supported Commands:
- `/help`: Show commands
- `/now`: Run pipeline immediately (checks lock)
- `/top`: Show top 5 bets from last run
- `/dryrun`: Simulate run (no alerts)
- `/status`: Show system version and health

## 5. Models
Models are stored in `models/`. See `audit_pack/A5_models/models_manifest.json` for hashes.
- `catboost_v1_latest.pkl` (Soccer)
- `catboost_basketball.cbm` (Basketball - Optional)

## 6. Logs & Audit
- **Predictions**: `audit_pack/A9_live/predictions.csv`
- **Sent Alerts**: `audit_pack/A9_live/alerts_sent.csv`
- **Exposure/ROI**: Run `python3 scripts/calculate_roi.py`
