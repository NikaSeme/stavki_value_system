from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from src.data.odds_api_client import fetch_odds
from src.data.odds_normalize import normalize_odds_events, best_price_by_outcome


def main():
    # defaults: NBA head-to-head in EU region (cheap)
    sport_key = os.getenv("ODDS_SPORT_KEY", "basketball_nba")
    regions = os.getenv("ODDS_REGIONS", "eu")
    markets = os.getenv("ODDS_MARKETS", "h2h")
    bookmakers = os.getenv("ODDS_BOOKMAKERS")  # optional

    now = datetime.now(timezone.utc)
    to_ = now + timedelta(hours=48)
    events = fetch_odds(
        sport_key=sport_key,
        regions=regions,
        markets=markets,
        bookmakers=bookmakers,
        commence_time_from=now.isoformat().replace("+00:00", "Z"),
        commence_time_to=to_.isoformat().replace("+00:00", "Z"),
    )
    rows = normalize_odds_events(events)
    best = best_price_by_outcome(rows)

    # print small sample
    print(f"events={len(events)} rows={len(rows)} best_rows={len(best)}")
    print(best[:5])


if __name__ == "__main__":
    main()
