# T310: Line Movement Tracking - User Guide

## Overview

Time-series odds tracking system for capturing market movements, calculating CLV (Closing Line Value), and detecting significant line shifts.

## Components

### 1. OddsTracker (`src/data/odds_tracker.py`)

**Core time-series storage with SQLite.**

**Database Tables:**
- `odds_history`: Time-series odds snapshots
- `closing_lines`: Final odds before match start
- `clv_tracking`: Bet tracking for CLV calculation

**Usage:**
```python
from src.data.odds_tracker import OddsTracker

tracker = OddsTracker()

# Store odds snapshot
odds_data = {
    'Pinnacle': {'home': 2.10, 'draw': 3.40, 'away': 3.50},
    'Bet365': {'home': 2.08, 'draw': 3.45, 'away': 3.55}
}
tracker.store_odds_snapshot(match_id, odds_data, is_opening=True)

# Get line movement stats
stats = tracker.calculate_movement_stats(match_id)
```

### 2. LineMovementFeatures (`src/features/line_movement_features.py`)

**12 new ML features from odds time-series.**

**Features:**
1-6. `home/draw/away_odds_open/current`
7-8. `home/away_odds_change_pct`
9. `sharp_move_detected` (> 10% in <12h)
10. `odds_volatility`
11. `time_to_match_hours`
12. `market_efficiency_score`

**Usage:**
```python
from src.features.line_movement_features import LineMovementFeatures

extractor = LineMovementFeatures()
features = extractor.extract_for_match(match_id, commence_time)
```

### 3. CLVCalculator (`src/analysis/clv_calculator.py`)

**Closing Line Value tracking.**

**Formula:**
```
CLV% = (bet_odds - closing_odds) / closing_odds * 100
```

- Positive CLV = Beat closing (good)
- Negative CLV = Worse than closing (bad)

**Usage:**
```python
from src.analysis.clv_calculator import CLVCalculator

calc = CLVCalculator()

# Track bet
calc.track_bet("bet_001", match_id, 'home', bet_odds=2.15)

# Get CLV after match
clv_data = calc.get_clv("bet_001")
print(f"CLV: {clv_data['clv_percent']:+.2f}%")

# Cumulative stats
stats = calc.get_cumulative_clv()
print(f"Average CLV: {stats['avg_clv']:+.2f}%")
```

### 4. LineMovementAlerts (`src/alerts/line_movement_alerts.py`)

**Alert triggers for significant movements.**

**Alert Types:**
- **Sharp Move:** >10% change in <12 hours
- **Discrepancy:** >15% difference between bookmakers  
- **Late Move:** >5% change within 2 hours of match

**Usage:**
```python
from src.alerts.line_movement_alerts import LineMovementAlerts

alerts_system = LineMovementAlerts()
alerts = alerts_system.check_all_alerts(match_id, commence_time, current_odds)

for alert in alerts:
    print(f"[{alert['type']}] {alert['message']}")
```

## Database Schema

**Location:** `data/odds/odds_timeseries.db`

```sql
-- Time-series odds
CREATE TABLE odds_history (
    match_id TEXT,
    timestamp INTEGER,
    bookmaker TEXT,
    outcome TEXT,
    odds REAL,
    is_opening BOOLEAN,
    is_closing BOOLEAN
);

-- Closing lines
CREATE TABLE closing_lines (
    match_id TEXT PRIMARY KEY,
    home_team TEXT,
    away_team TEXT,
    home_odds_open REAL,
    home_odds_close REAL,
    ...
);

-- CLV tracking
CREATE TABLE clv_tracking (
    bet_id TEXT PRIMARY KEY,
    match_id TEXT,
    bet_odds REAL,
    closing_odds REAL,
    clv_percent REAL
);
```

## Testing

```bash
# Test tracker
python src/data/odds_tracker.py

# Test line features
python scripts/test_line_features.py

# Test CLV
python -c "from src.analysis.clv_calculator import test_clv; test_clv()"

# Test alerts
python -c "from src.alerts.line_movement_alerts import test_alerts; test_alerts()"
```

## Example Workflow

### 1. Track Match Odds

```python
tracker = OddsTracker()
match_id = "epl_20260125_001"

# Store opening odds (first fetch)
opening = fetch_odds_from_api()
tracker.store_odds_snapshot(match_id, opening, is_opening=True)

# Poll periodically (every 15 min)
while hours_to_match > 0:
    current = fetch_odds_from_api()
    tracker.store_odds_snapshot(match_id, current)
    time.sleep(900)  # 15 min

# Store closing (final fetch)
closing = fetch_odds_from_api()
tracker.store_odds_snapshot(match_id, closing, is_closing=True)
```

### 2. Place Bet with CLV Tracking

```python
calc = CLVCalculator()

# Place bet
bet_odds = 2.15
calc.track_bet("bet_456", match_id, 'home', bet_odds)

# After match closes
clv = calc.get_clv("bet_456")
if clv['beat_closing']:
    print(f"✅ Beat closing by {clv['clv_percent']:+.1f}%")
else:
    print(f"❌ Closed {abs(clv['clv_percent']):.1f}% better")
```

### 3. Monitor Alerts

```python
alerts_system = LineMovementAlerts()

# Check before betting
alerts = alerts_system.check_all_alerts(match_id, commence_time, current_odds)

if any(a['type'] == 'sharp_move' for a in alerts):
    print("🚨 Sharp move detected - investigate before betting!")
```

## Performance Metrics

### Test Results

**Line Movement Detection:**
- Sharp move: -13.64% detected ✓
- Volatility: 0.1333 calculated
- Time to match: Accurate
- Market efficiency: 0.9240 (typical)

**CLV Calculation:**
- Test bet @ 2.05, closing @ 1.90
- CLV: +7.89% ✓
- Beat closing line validated

**Alerts:**
- Sharp move: Triggered correctly
- Discrepancy: 15%+ detected
- Late move: <2h detection working

## Database Maintenance

### Archival Strategy

```python
# Archive old matches (>30 days)
conn = sqlite3.connect('data/odds/odds_timeseries.db')
conn.execute('''
    DELETE FROM odds_history 
    WHERE timestamp < ?
''', (int(time.time()) - 30*24*3600,))
conn.commit()
```

### Database Size

- ~1KB per match per bookmaker per snapshot
- 15-min polling = 96 snapshots/day
- 10 matches/day × 3 bookmakers × 96 snapshots = 2.8MB/day
- Monthly archival keeps DB <100MB

## Integration with ML Pipeline

**Next Steps (Phase 4):**
1. Add line features to live extractor
2. Backfill historical data with open/close
3. Retrain model with 40 features (28 current + 12 line)
4. Validate improved predictions

## Alerts Log

**Location:** `data/logs/line_movement_alerts.json`

```json
[
  {
    "timestamp": 1737598234,
    "match_id": "epl_001",
    "type": "sharp_move",
    "severity": "high",
    "change_pct": -13.6,
    "message": "Sharp move detected: home odds moved 13.6% in 12h"
  }
]
```

## Files Created

- `src/data/odds_tracker.py` - Core tracker
- `src/features/line_movement_features.py` - Feature extractor
- `src/analysis/clv_calculator.py` - CLV module
- `src/alerts/line_movement_alerts.py` - Alert system
- `scripts/test_line_features.py` - Test script
- `data/odds/odds_timeseries.db` - Time-series DB

## Best Practices

1. **Poll Frequency:** 15 min is optimal (respects API limits)
2. **Closing Line:** Fetch 5 min before match start
3. **CLV Threshold:** Aim for >2% average CLV
4. **Sharp Moves:** Investigate before betting
5. **Database:** Archive monthly to keep size manageable

## Troubleshooting

**Q: No closing odds captured**
- Ensure polls continue until match start
- Schedule final fetch at commence_time - 5 min

**Q: CLV always negative**
- Check bet timing (bet earlier for better odds)
- Review line movement patterns
- Consider sources with slower lines

**Q: Database growing too large**
- Run monthly archival script
- Reduce poll frequency for distant matches
- Delete odds_history for old matches (keep closing_lines)

---

**Status:** Core modules complete (Phases 1-5)  
**Next:** Model retraining with line features (Phase 7)
