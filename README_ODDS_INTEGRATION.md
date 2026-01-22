# PLAN 1 – Odds API integration starter

This pack adds a clean, minimal integration with The Odds API:
- Fetch odds (v4)
- Normalize JSON to a flat table of outcomes
- Compute implied probabilities + simple no-vig (proportional)
- Pick best odds across bookmakers
- Compute EV given model probabilities

## 1) Environment variables
Create `.env` (or export vars) with:
- ODDS_API_KEY=...
- ODDS_API_BASE=https://api.the-odds-api.com

## 2) Quick test (CLI-less)
python -m src.pipeline.fetch_and_normalize_odds

It will print a small sample dataframe-like dict list.

## 3) How you plug into your project
- Keep `src/data/odds_api_client.py` and `src/data/odds_normalize.py`
- In your pipeline, call `fetch_odds()` then `normalize_odds_events()` and persist
- When you have model probabilities per event, call `compute_value_bets_1x2()`

Notes:
- This code is intentionally conservative about request costs: you control markets and regions.
- No-vig here is the simple proportional method. You can replace with Shin later.
