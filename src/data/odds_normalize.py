from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


def _to_iso(dt: str) -> str:
    # already ISO in Odds API if dateFormat=iso; normalize to Z
    try:
        x = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        return x.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return dt


def normalize_odds_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten Odds API event JSON into outcome rows.

    Row schema:
    - event_id, sport_key, commence_time
    - home_team, away_team
    - bookmaker_key, bookmaker_title, last_update
    - market_key (e.g., h2h), outcome_name, outcome_price
    """
    rows: List[Dict[str, Any]] = []
    for ev in events:
        event_id = ev.get("id")
        sport_key = ev.get("sport_key")
        commence_time = _to_iso(ev.get("commence_time"))
        home_team = ev.get("home_team")
        away_team = ev.get("away_team")

        for bk in ev.get("bookmakers", []) or []:
            bk_key = bk.get("key")
            bk_title = bk.get("title")
            last_update = _to_iso(bk.get("last_update")) if bk.get("last_update") else None

            for m in bk.get("markets", []) or []:
                market_key = m.get("key")
                for out in m.get("outcomes", []) or []:
                    rows.append(
                        {
                            "event_id": event_id,
                            "sport_key": sport_key,
                            "commence_time": commence_time,
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker_key": bk_key,
                            "bookmaker_title": bk_title,
                            "last_update": last_update,
                            "market_key": market_key,
                            "outcome_name": out.get("name"),
                            "outcome_price": out.get("price"),
                        }
                    )
    return rows


def best_price_by_outcome(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pick the best (max) decimal price per event_id+market_key+outcome_name."""
    best: Dict[tuple, Dict[str, Any]] = {}
    for r in rows:
        key = (r["event_id"], r["market_key"], r["outcome_name"])
        price = r.get("outcome_price")
        if price is None:
            continue
        cur = best.get(key)
        if cur is None or float(price) > float(cur["outcome_price"]):
            best[key] = r
    return list(best.values())
