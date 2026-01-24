# Deploying to Google Compute Engine (GCE)

This guide walks you through setting up the betting system on a cloud server.

## 1. Create VM Instance

1. Go to **Google Cloud Console** > **Compute Engine** > **VM Instances**.
2. Click **Create Instance**.
3. **Settings:**
   - **Name:** `stavki-bot`
   - **Region:** `us-east1` (or closest to you)
   - **Architecture:** `x86/Intel` (Required)
   - **Machine Type:** `e2-medium` (2 vCPU, 4GB RAM) - **Recommended** for Python/ML work.
   - **Boot Disk:** `30-50 GB` Balanced Persistent Disk (SSD)
4. **Firewall:** Allow HTTP/HTTPS (optional, not strictly needed for this bot).
5. Click **Create**.

## 2. Connect to Server

1. Click **SSH** button next to your instance in Cloud Console.
2. A terminal window will open.

## 3. Install & Setup

Run these commands in the SSH terminal:

### A. Clone Repository
*(Method 1: HTTPS - requires typing username/token)*
```bash
git clone https://github.com/serni13678-alt/stavki_value_system.git
cd stavki_value_system
```

*(Method 2: Upload Zip)*
- Use the "Upload File" button in Google SSH window to upload a zip of your code.
- Unzip: `unzip stavki_value_system.zip && cd stavki_value_system`

### B. Run Setup Script
```bash
chmod +x scripts/setup_gce.sh
./scripts/setup_gce.sh
```

### C. Configure Environment
```bash
nano .env
```
- Paste your API keys (`ODDS_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) inside.
- Press `Ctrl+X`, then `Y`, then `Enter` to save.

## 4. Run Manually (Testing)

Test if everything works:
```bash
source venv/bin/activate
python run_scheduler.py --telegram --max-runs 1
```

## 5. Setup Automatic Background Execution

### Option A: Systemd Service (Robust - Recommended)
Keeps the scheduler running 24/7 and restarts it if it crashes.

1. **Edit the service file template:**
   ```bash
   nano scripts/stavki-scheduler.service
   ```
   - Change `User=username` to your GCE username (run `whoami` to check).
   - Update paths if different from `/home/username/stavki_value_system`.

2. **Install Service:**
   ```bash
   sudo cp scripts/stavki-scheduler.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable stavki-scheduler
   sudo systemctl start stavki-scheduler
   ```

3. **Check Status:**
   ```bash
   sudo systemctl status stavki-scheduler
   ```

4. **View Logs:**
   ```bash
   tail -f outputs/logs/scheduler_service.log
   ```

### Option B: Cron Jobs
Use this for the **Daily Summary** (runs once a day).

1. Open crontab:
   ```bash
   crontab -e
   ```

2. Add this line (replace `/home/username/...` with your real path):
   ```bash
   # Run Daily Summary at 8:00 AM UTC
   0 8 * * * cd /home/username/stavki_value_system && ./venv/bin/python scripts/send_daily_summary.py >> outputs/logs/cron_daily.log 2>&1
   ```

## 6. Maintenance

- **Update Code:**
  ```bash
  cd stavki_value_system
  git pull
  sudo systemctl restart stavki-scheduler
  ```

- **Monitor Costs:**
  - `e2-micro` is often free in specific regions (us-east1, us-west1, us-central1).
  - Check Google Cloud Billing to avoid surprises.
