# Data Sources & Schema

## 1. Data Sources
| Dataset | Source | Purpose | Location |
|---|---|---|---|
| **Historical Matches** | football-data.co.uk | Training/Backtesting | `data/raw/` |
| **Historical Odds** | football-data.co.uk | Market Logic/EV | `data/raw/` |
| **Live Odds** | The Odds API | Live Inference | API (Not stored permanently) |

## 2. Feature Schema (Processed)
See `data_sample.csv` for actual values.

| Column | Type | Description |
|---|---|---|
| `Date` | DateTime | Match date (UTC) |
| `HomeTeam`/`AwayTeam` | String | Standardized team names |
| `HomeEloBefore` | Float | Elo rating before match |
| `HomePointsL5` | Int | Points in last 5 games |
| `MarketProbHomeNoVig` | Float | Bookmaker implied prob (vig removed) |
| `OddsHomeAwayRatio` | Float | Market bias indicator |
| `FTR` | String | Full Time Result (H/D/A) - **TARGET** |

## 3. Storage
- **Format:** CSV (local), potentially DB (future).
- **Sensitive Data:** None (Public sports data).
