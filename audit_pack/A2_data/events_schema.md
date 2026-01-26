# Unified Event Schema (v3.3)

Standardized format for multi-sport events (Soccer, Basketball, etc.) to enable a single value-finding pipeline.

## Parquet Schema (`events_latest.parquet`)

| Column | Type | Description | Example |
|---|---|---|---|
| `event_id` | String | Unique ID from Odds API | `984539824` |
| `sport` | String | Sport category (normalized) | `soccer`, `basketball` |
| `league` | String | League key | `soccer_epl`, `basketball_nba` |
| `home_team` | String | Normalized home team name | `Arsenal`, `LA Lakers` |
| `away_team` | String | Normalized away team name | `Chelsea`, `Boston Celtics` |
| `start_time_utc` | Timestamp | Match start time (UTC) | `2024-05-20T19:00:00Z` |
| `bookmaker` | String | Bookmaker title | `Pinnacle`, `Bet365` |
| `odds_snapshot_time` | Timestamp | Time odds were fetched | `2024-05-20T10:00:00Z` |
| `market_type` | String | Market category | `1x2`, `h2h`, `spreads`, `totals` |
| `selection` | String | Outcome selection | `HOME`, `DRAW`, `AWAY` (Soccer)<br>`HOME`, `AWAY` (Basket H2H) |
| `odds_decimal` | Float | Decimal odds | `2.10` |
| `point` | Float | Spread/Total point (optional) | `2.5` (Total), `-5.5` (Spread) |

## Mapping Rules

### Soccer
- **Source Market**: `h2h`
- **Target Market**: `1x2`
- **Selections**:
  - `Home Team Name` -> `HOME`
  - `Away Team Name` -> `AWAY`
  - `Draw` -> `DRAW`

### Basketball
- **Source Market**: `h2h` (Moneyline)
- **Target Market**: `moneyline`
- **Selections**:
  - `Home Team Name` -> `HOME`
  - `Away Team Name` -> `AWAY`

- **Source Market**: `spreads`
- **Target Market**: `spread`
- **Selections**:
  - `Home Team Name` -> `HOME`
  - `Away Team Name` -> `AWAY`
  - `point` preserved for handicap

- **Source Market**: `totals`
- **Target Market**: `total`
- **Selections**:
  - `Over` -> `OVER`
  - `Under` -> `UNDER`
  - `point` preserved for total line
