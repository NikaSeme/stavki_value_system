# API & Data Pipeline Audit

> **Date**: 2026-02-01  
> **Scope**: Evidence-based documentation of current behavior (NO CODE CHANGES)

---

## 1. What Exactly Do We Fetch from the Odds API?

### Endpoints Used

| Endpoint | Purpose | File |
|----------|---------|------|
| `GET /v4/sports/` | List available sports | `src/data/odds_api_client.py:53-62` |
| `GET /v4/sports/{sport_key}/odds/` | Fetch odds for events | `src/data/odds_api_client.py:65-100` |

### Request Parameters

**Evidence** — from [odds_api_client.py](file:///Users/macuser/Documents/something/stavki_value_system/src/data/odds_api_client.py#L65-100):

```python
def fetch_odds(
    sport_key: str,
    regions: str,
    markets: str,
    *,
    odds_format: str = "decimal",
    date_format: str = "iso",
    bookmakers: Optional[str] = None,
    commence_time_from: Optional[str] = None,
    commence_time_to: Optional[str] = None,
    cfg: Optional[OddsAPIConfig] = None,
) -> List[Dict[str, Any]]:
```

**Actual call** — from [run_odds_pipeline.py](file:///Users/macuser/Documents/something/stavki_value_system/scripts/run_odds_pipeline.py#L143-151):

```python
events = fetch_odds(
    sport_key=sport_key,
    regions=args.regions,    # default: "eu,uk,us"
    markets=args.markets,    # default: "h2h"
    odds_format=args.odds_format,  # default: "decimal"
    commence_time_from=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    commence_time_to=to_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    cfg=api_config
)
```

### Raw JSON Sample — Events List

**Source**: `outputs/odds/raw_soccer_epl_20260122_235753.json`

```json
[
  {
    "id": "a396966205b21287109ea8cae42e44c1",
    "sport_key": "soccer_epl",
    "sport_title": "EPL",
    "commence_time": "2026-01-24T12:30:00Z",
    "home_team": "West Ham United",
    "away_team": "Sunderland",
    "bookmakers": [ ... ]
  }
]
```

### Raw JSON Sample — Odds for 1 Event (1 Bookmaker)

```json
{
  "key": "pinnacle",
  "title": "Pinnacle",
  "last_update": "2026-01-22T22:57:19Z",
  "markets": [
    {
      "key": "h2h",
      "last_update": "2026-01-22T22:57:19Z",
      "outcomes": [
        { "name": "Sunderland", "price": 2.96 },
        { "name": "West Ham United", "price": 2.58 },
        { "name": "Draw", "price": 3.25 }
      ]
    }
  ]
}
```

---

## 2. How Do We Normalize Odds?

### Normalized Schema

**Evidence** — from [odds_normalize.py](file:///Users/macuser/Documents/something/stavki_value_system/src/data/odds_normalize.py#L16-64):

```python
def normalize_odds_events(events, snapshot_time=None):
    """
    Row schema:
    - event_id, sport_key, commence_time
    - home_team, away_team
    - bookmaker_key, bookmaker_title, last_update
    - market_key (e.g., h2h), outcome_name, outcome_price
    - odds_snapshot_time (ingestion time)
    """
```

### Normalization Example (1 Event)

**Raw outcomes** (1xBet for West Ham vs Sunderland):
```
Sunderland:       3.0
West Ham United:  2.6
Draw:             3.37
```

**Normalized rows** (3 rows for this bookmaker):

| event_id | home_team | away_team | bookmaker_key | market_key | outcome_name | outcome_price |
|----------|-----------|-----------|---------------|------------|--------------|---------------|
| a396966... | West Ham United | Sunderland | onexbet | h2h | Sunderland | 3.0 |
| a396966... | West Ham United | Sunderland | onexbet | h2h | West Ham United | 2.6 |
| a396966... | West Ham United | Sunderland | onexbet | h2h | Draw | 3.37 |

### Home/Draw/Away Mapping

**Evidence** — from [live_extractor.py](file:///Users/macuser/Documents/something/stavki_value_system/src/features/live_extractor.py#L240-242):

```python
home_odds_list = event_odds[event_odds['outcome_name'] == home_team]['outcome_price']
draw_odds_list = event_odds[event_odds['outcome_name'] == 'Draw']['outcome_price']
away_odds_list = event_odds[event_odds['outcome_name'] == away_team]['outcome_price']
```

**Mapping Rule**:
- `outcome_name == home_team` → Home
- `outcome_name == "Draw"` → Draw
- `outcome_name == away_team` → Away

---

## 3. Are We Using 1X2 Market Correctly?

### Market Key Used

**Evidence** — grep for `h2h`:

| Location | Usage |
|----------|-------|
| `run_odds_pipeline.py:51` | `default='h2h'` |
| `value_live.py:484` | `market_key: str = 'h2h'` |
| `value.py:16` | `market_key: str = "h2h"` |

**Confirmation**: Market key is `h2h` (Odds API terminology for 1X2/Moneyline)

### Outcome Name Mapping Rules

**Evidence** — from [value_live.py](file:///Users/macuser/Documents/something/stavki_value_system/src/strategy/value_live.py#L76-112):

```python
def validate_outcome_mapping(outcome_name, home_team, away_team, model_probs):
    # Normalize all names
    norm_outcome = normalize_team_name(outcome_name)
    norm_home = normalize_team_name(home_team)
    norm_away = normalize_team_name(away_team)
    
    # Check for draw
    if outcome_name.lower().strip() == 'draw':
        return model_probs.get('Draw')
    
    # Check if outcome matches home or away
    if norm_outcome == norm_home:
        return model_probs.get(home_team)
    elif norm_outcome == norm_away:
        return model_probs.get(away_team)
```

### Edge Case Handling

#### Missing Draw

**Behavior**: No explicit handling. If `outcome_name == 'Draw'` not found, `draw_odds_list` will be empty.

**Evidence** — from [live_extractor.py](file:///Users/macuser/Documents/something/stavki_value_system/src/features/live_extractor.py#L244-246):

```python
draw_odds = draw_odds_list.mean() if len(draw_odds_list) > 0 else 3.5  # Default fallback
```

**Verdict**: Falls back to default 3.5 if missing.

#### "Tie" vs "Draw" Naming

**Behavior**: Only `"Draw"` (exact, case-insensitive) is recognized.

**Evidence**:
```python
if outcome_name.lower().strip() == 'draw':
```

**Risk**: If API returns `"Tie"`, it would NOT be recognized as draw. Would be treated as unknown outcome.

#### Duplicate Events

**Behavior**: No explicit deduplication. If same event_id appears twice, both are processed.

**Evidence**: No deduplication code found in `normalize_odds_events` or `best_price_by_outcome`.

---

## 4. Odds Selection Logic

### Mode: Single Book vs Best-Of

**Evidence** — from [odds_normalize.py](file:///Users/macuser/Documents/something/stavki_value_system/src/data/odds_normalize.py#L67-80):

```python
def best_price_by_outcome(rows, mode: str = 'single_book'):
    """
    Modes:
    - 'best_of': Max price per outcome across all books
    - 'single_book': Pick ONE bookmaker per event
    
    For 'single_book', we pick the bookmaker with highest 'Home' odds.
    """
```

### Default Mode: `single_book`

**Selection Rule** — from [odds_normalize.py](file:///Users/macuser/Documents/something/stavki_value_system/src/data/odds_normalize.py#L126-142):

```python
# Find best book by max Home odds
for bk, items in books.items():
    for item in items:
        if item.get("outcome_name") == item.get("home_team"):
            home_price = float(item["outcome_price"])
            break
    if home_price > max_home_price:
        max_home_price = home_price
        best_bk_key = bk
```

**Behavior**: Selects the bookmaker with highest Home odds, then uses ALL outcomes from that single bookmaker.

### Example

If for West Ham vs Sunderland:
- 1xBet: Home=2.6, Draw=3.37, Away=3.0
- Pinnacle: Home=2.58, Draw=3.25, Away=2.96
- Unibet: Home=2.65, Draw=3.25, Away=2.95

**Selection**: Unibet wins (Home=2.65 is highest)  
**Output**: Uses Unibet's entire line: Home=2.65, Draw=3.25, Away=2.95

### Best-Of Mode (Alternative)

When `mode='best_of'`:
- Home: 2.65 (Unibet)
- Draw: 3.37 (1xBet)
- Away: 3.0 (1xBet)

**Note**: This creates a synthetic line that may not be achievable at any single bookmaker.

---

## 5. Timing & Freshness

### When Do We Fetch Odds?

**Evidence** — from [run_odds_pipeline.py](file:///Users/macuser/Documents/something/stavki_value_system/scripts/run_odds_pipeline.py#L65-69):

```python
parser.add_argument(
    '--hours-ahead',
    type=int,
    default=48,
    help='Fetch events starting within N hours from now'
)
```

**Default**: 48 hours ahead of current time

### Collected_at Timestamps

**Yes** — stored as `odds_snapshot_time`

**Evidence** — from [odds_normalize.py](file:///Users/macuser/Documents/something/stavki_value_system/src/data/odds_normalize.py#L28-30):

```python
if snapshot_time is None:
    snapshot_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
```

**Schema confirms**: `odds_snapshot_time` field in normalized output

### Timezone Conversions

**Evidence** — from [odds_normalize.py](file:///Users/macuser/Documents/something/stavki_value_system/src/data/odds_normalize.py#L7-13):

```python
def _to_iso(dt: str) -> str:
    try:
        x = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        return x.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return dt
```

**Behavior**: All times normalized to UTC with "Z" suffix.

### Sample Timing from Logs

From raw JSON:
- `commence_time`: `"2026-01-24T12:30:00Z"` (kickoff)
- `last_update`: `"2026-01-22T22:57:19Z"` (odds update time)
- Fetch time: `2026-01-22T23:57:53Z` (filename timestamp)

**Hours before kickoff**: ~36 hours
