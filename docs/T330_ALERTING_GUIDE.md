# T330: Alerting & Monitoring Guide

## Overview

Real-time alerting system with Telegram notifications and comprehensive performance tracking.

## Components

### 1. Telegram Bot (`src/alerts/telegram_bot.py`)

Sends alerts via Telegram.

**Setup:**
1. Create bot: Talk to [@BotFather](https://t.me/botfather)
2. Get token: `/newbot` → follow prompts
3. Get chat ID: Talk to [@userinfobot](https://t.me/userinfobot)
4. Add to `.env`:
```bash
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

**Test:**
```bash
python src/alerts/telegram_bot.py
```

### 2. Alert Manager (`src/alerts/alert_manager.py`)

Central alert coordination.

**Alert Types:**
- 🎯 **Value Bet:** EV > 5% threshold
- 📊 **Line Move:** Sharp move >10% in <12h
- ⚠️ **Performance:** Drawdown >20%
- 🔧 **System Health:** Component failures

**Usage:**
```python
from src.alerts.alert_manager import AlertManager

manager = AlertManager()

# Send value bet alert
manager.send_value_bet_alert({
    'match': 'Man City vs Liverpool',
    'market': 'Home Win',
    'odds': 2.15,
    'model_prob': 0.523,
    'ev': 6.7,
    'stake': 50.0
})
```

### 3. Performance Monitor (`src/monitoring/performance_monitor.py`)

Tracks all bets and metrics.

**Metrics:**
- ROI (Return on Investment)
- Hit Rate / Win %
- CLV (Closing Line Value)
- Max Drawdown
- Recent Record

**Usage:**
```python
from src.monitoring.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()

# Track bet
monitor.track_bet(
    bet_id='bet_001',
    match_name='City vs Pool',
    outcome='home',
    odds=2.10,
    stake=100,
    result='win',  # 'win', 'loss', 'push'
    ev_percent=5.0,
    clv_percent=3.5
)

# Get summary
summary = monitor.get_performance_summary()
print(f"ROI: {summary['roi']:+.1f}%")
```

### 4. Daily Summary (`scripts/send_daily_summary.py`)

Automated daily reports.

**Setup with cron:**
```bash
# Run at midnight daily
0 0 * * * cd /path/to/project && venv/bin/python scripts/send_daily_summary.py
```

**Manual run:**
```bash
python scripts/send_daily_summary.py
```

## Configuration

**File:** `config/alerts_config.json`

```json
{
  "thresholds": {
    "min_ev_percent": 5.0,      // Min EV for alerts
    "max_drawdown_percent": 20.0,  // Max drawdown before alert
    "sharp_move_pct": 10.0      // Line move threshold
  },
  "value_bets": {
    "enabled": true,
    "min_confidence": 0.60
  }
}
```

## Alert Examples

### Value Bet Alert
```
🎯 VALUE BET DETECTED

Match: Man City vs Liverpool
Market: Home Win
Current Odds: 2.15
Model Prob: 52.3%
EV: +6.7%
Recommended Stake: £50.00

⚡ Act fast - odds may change
```

### Line Movement Alert
```
📉 SHARP LINE MOVE

Match: Chelsea vs Arsenal
Outcome: Home
Movement: 2.50 → 2.10 (-16.0%)
Timeframe: 6 hours

🔍 Investigate before betting
```

### Performance Alert
```
⚠️ PERFORMANCE WARNING

Current Drawdown: 22.5%
Threshold: 20.0%

Recent Performance:
- Last 10 bets: 3W-7L
- Week ROI: -5.2%
- Week CLV: -1.8%

📊 Consider reviewing strategy
```

### Daily Summary
```
📊 DAILY SUMMARY - 2026-01-23

Today:
Bets: 3
Wins: 2 (66.7%)
Profit: £+45.00
ROI: +15.0%

This Week:
Bets: 15
ROI: +8.5%
CLV: +2.1%

All-Time:
Total Bets: 125
Hit Rate: 58.4%
ROI: +6.2%

📈 Keep tracking!
```

## Testing

### Test All Alerts
```bash
python src/alerts/alert_manager.py
```

Sends 4 test alerts to Telegram.

### Test Performance Monitor
```bash
python src/monitoring/performance_monitor.py
```

Simulates 5 bets and shows summary.

## Troubleshooting

**No alerts received:**
1. Check `TELEGRAM_BOT_TOKEN` set in `.env`
2. Check `TELEGRAM_CHAT_ID` set correctly
3. Test bot: `python src/alerts/telegram_bot.py`
4. Check bot has permission to message you

**Duplicate alerts:**
- Alerts deduplicated within 60 min window
- Check `data/logs/alert_history.json`

**Performance metrics wrong:**
- Check database: `data/performance.db`
- Verify bet tracking calls
- Run test: `python src/monitoring/performance_monitor.py`

## Integration

### With Value Finder

```python
from src.alerts.alert_manager import AlertManager

manager = AlertManager()

# In value finder loop
if ev > threshold:
    manager.send_value_bet_alert({
        'match': f"{home} vs {away}",
        'market': market,
        'odds': odds,
        'model_prob': model_prob,
        'ev': ev,
        'stake': recommended_stake
    })
```

### With Line Tracker (T310)

```python
from src.alerts.alert_manager import AlertManager
from src.data.odds_tracker import OddsTracker

manager = AlertManager()
tracker = OddsTracker()

# Check for movements
stats = tracker.calculate_movement_stats(match_id)

if abs(stats['home']['change_pct']) > 10:
    manager.send_line_move_alert({
        'match': match_name,
        'outcome': 'home',
        'from_odds': stats['home']['open'],
        'to_odds': stats['home']['current'],
        'change_pct': stats['home']['change_pct'],
        'hours': 12
    })
```

## Database

**Location:** `data/performance.db`

**Tables:**
- `bets` - All tracked bets
- `performance_snapshots` - Daily snapshots

**Query examples:**
```bash
sqlite3 data/performance.db "SELECT * FROM bets ORDER BY timestamp DESC LIMIT 10"
sqlite3 data/performance.db "SELECT COUNT(*), SUM(profit) FROM bets WHERE result='win'"
```

## Security

**Protect credentials:**
- Never commit `.env` to Git
- `.env` already in `.gitignore`
- Use environment variables only

**Telegram security:**
- Bot token is secret (like password)
- Chat ID identifies recipient
- Can restrict bot to specific users

## Files Created

- `src/alerts/telegram_bot.py` - Telegram integration
- `src/alerts/alert_manager.py` - Alert coordination
- `src/monitoring/performance_monitor.py` - Performance tracking
- `scripts/send_daily_summary.py` - Daily reports
- `config/alerts_config.json` - Configuration
- `docs/T330_ALERTING_GUIDE.md` - This guide

## Status

✅ **Phase 1-5 Complete**
- Telegram bot integrated
- All alert types working
- Performance monitoring active
- Daily summaries ready

**Ready for production!** 🎯
