# BLOCKERS & RISKS (v5.0)

## 1. Critical Risks (Production)
- **API Rate Limits**: The OddsAPI plan allows 500 req/mo. Current schedule (2/day * 6 leagues) = ~360 req/mo. Safe, but adding hourly runs will break it.
- **Model Drift**: Soccer model trained on 2023-2025. Verify performance monthly.
- **Telegram Ban**: Sending >20 msg/sec can trigger ban. We send 1 consolidated table (Safe).

## 2. Missing Features (Planned v5.1)
- **Basketball Model**: Currently fallback or disabled. Needs dedicated CatBoost training.
- **Live Line Movement (Model C)**: tracking odds shift in last 10 mins. Not implemented.
- **Web Dashboard**: Currently only CLI/CSV/Telegram.

## 3. Technical Debt
- `run_value_finder.py` is large (~700 lines). Split into `src/pipelines/run.py`.
- Hardcoded league lists in `MAJOR_LEAGUES` constant. Move to `configs/leagues.json`.
