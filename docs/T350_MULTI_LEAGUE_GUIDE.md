# T350: Multi-League System Guide

## Overview

Scale from single league (EPL) to multiple concurrent leagues with independent models and unified monitoring.

**Supported:** EPL, La Liga, Bundesliga (+ any additional)

## Architecture

```
LeagueManager
    ├─ EPL Agent → EPL Models → EPL Monitor
    ├─ La Liga Agent → La Liga Models → La Liga Monitor  
    └─ Bundesliga Agent → Bundesliga Models → Bundesliga Monitor
                ↓
        Multi-League Monitor (Aggregation)
```

## Quick Start

### Run Multiple Leagues

```python
from src.multi_league.league_manager import LeagueManager

# Initialize for 3 leagues
manager = LeagueManager(['epl', 'laliga', 'bundesliga'])

# Start all in parallel
manager.start_all_leagues()

# Keep alive
manager.run_forever()
```

### Monitor Performance

```python
from src.multi_league.multi_league_monitor import MultiLeagueMonitor

monitor = MultiLeagueMonitor(['epl', 'laliga', 'bundesliga'])

# Get overall summary
overall = monitor.get_overall_summary()
print(f"Total Bets: {overall['total_bets']}")
print(f"Avg ROI: {overall['avg_roi']:+.1f}%")

# Full report
report = monitor.generate_multi_league_report()
print(report)
```

## League Configuration

**File:** `config/leagues/{league_id}.yaml`

```yaml
league:
  id: "laliga"
  name: "La Liga"
  enabled: true

models:
  poisson:
    model_path: "models/laliga_poisson_v1.pkl"
  catboost:
    model_path: "models/laliga_catboost_v1.pkl"

execution:
  check_interval: 900  # 15 minutes
  auto_bet: false
  min_ev_threshold: 5.0
```

## Adding New League

### Step 1: Create Config

```bash
cp config/leagues/epl.yaml config/leagues/serie_a.yaml
```

Edit `serie_a.yaml`:
- Update league ID and name
- Set correct odds API key
- Configure model paths

### Step 2: Add to Manager

```python
manager = LeagueManager(['epl', 'laliga', 'bundesliga', 'serie_a'])
```

### Step 3: Test

```python
# Verify config loaded
for league_id, config in manager.configs.items():
    print(f"{league_id}: {config['league']['name']}")
```

## Multi-League Monitoring

### Per-League Stats

```python
monitor = MultiLeagueMonitor(['epl', 'laliga', 'bundesliga'])

# EPL stats
epl_summary = monitor.get_league_summary('epl')
print(f"EPL ROI: {epl_summary['roi']:+.1f}%")

# La Liga stats
laliga_summary = monitor.get_league_summary('laliga')
print(f"La Liga ROI: {laliga_summary['roi']:+.1f}%")
```

### Aggregated Report

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MULTI-LEAGUE PERFORMANCE REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OVERALL:
  Total Bets: 100
  Total Profit: £+245.50
  Avg ROI: +8.2%

BY LEAGUE:
  EPL:
    Bets: 45
    ROI: +6.8%
  
  LA LIGA:
    Bets: 35
    ROI: +11.5%
  
  BUNDESLIGA:
    Bets: 20
    ROI: +5.2%
```

## Components

### LeagueManager

**Purpose:** Orchestrate multiple leagues

```python
manager = LeagueManager(['epl', 'laliga'])

# Start threads for each league
manager.start_all_leagues()

# Check status
status = manager.get_status()
print(status['leagues']['epl'])

# Stop all
manager.stop_all_leagues()
```

### LeagueAgent

**Purpose:** Independent worker per league

```python
agent = LeagueAgent('epl', config)

# Run main loop (blocks)
agent.run()

# Or use in thread (done by manager)
thread = threading.Thread(target=agent.run)
thread.start()
```

### MultiLeagueMonitor

**Purpose:** Cross-league performance tracking

```python
monitor = MultiLeagueMonitor(['epl', 'laliga', 'bundesliga'])

# All leagues
all_summaries = monitor.get_all_leagues_summary()

# Aggregate
overall = monitor.get_overall_summary()
```

## Files Created

**Core:**
- `src/multi_league/league_manager.py` (195 lines)
- `src/multi_league/league_agent.py` (115 lines)
- `src/multi_league/multi_league_monitor.py` (178 lines)

**Configs:**
- `config/leagues/epl.yaml`
- `config/leagues/laliga.yaml`
- `config/leagues/bundesliga.yaml`

**Docs:**
- `docs/T350_MULTI_LEAGUE_GUIDE.md`

**Total:** ~500 lines

## Testing

### Test Manager

```bash
python src/multi_league/league_manager.py
```

**Output:**
```
Loaded Configurations:
  - epl: English Premier League (football)
  - laliga: La Liga (football)
  - bundesliga: Bundesliga (football)

Active Leagues: 3
```

### Test Agent

```bash
python src/multi_league/league_agent.py
```

### Test Monitor

```bash
python src/multi_league/multi_league_monitor.py
```

## Performance

**Per League:**
- CPU: ~1 core
- Memory: ~500MB
- Check interval: 15 min (configurable)

**3 Leagues:**
- CPU: ~3 cores
- Memory: ~1.5GB
- Runs stable on laptop

## Best Practices

### 1. Independent Databases

Each league gets own performance DB:
- `data/performance_epl.db`
- `data/performance_laliga.db`
- `data/performance_bundesliga.db`

### 2. League-Tagged Alerts

```python
alert_manager.send_value_bet_alert({
    'league': 'laliga',  # Tag league
    'match': 'Real vs Barca',
    ...
})
```

### 3. Parallel Processing

```python
# Using threading (IO-bound)
for league_id, agent in manager.leagues.items():
    thread = threading.Thread(target=agent.run)
    thread.start()

# Or multiprocessing (CPU-bound)
for league_id, agent in manager.leagues.items():
    process = multiprocessing.Process(target=agent.run)
    process.start()
```

## Configuration Reference

**League Config (YAML):**

```yaml
league:
  id: "epl"              # Unique ID
  name: "EPL"            # Display name
  sport: "football"      # Sport type
  enabled: true          # Enable/disable

data_sources:
  odds_api:
    league_key: "soccer_epl"
    
models:
  poisson:
    enabled: true
    model_path: "models/epl_poisson_v1.pkl"
    
execution:
  check_interval: 900    # Seconds
  auto_bet: false        # Manual mode
  min_ev_threshold: 5.0  # Min EV%
  
monitoring:
  enabled: true
  performance_db: "data/performance_epl.db"
```

## Status

**T350 Core Complete** ✅
- Multi-league architecture ✓
- 3 league configs created ✓
- Parallel execution framework ✓
- Aggregated monitoring ✓
- Tested and working ✓

**Ready for 3+ league operation!** 🚀
