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


def normalize_odds_events(
    events: List[Dict[str, Any]], 
    snapshot_time: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Flatten Odds API event JSON into outcome rows.

    Row schema:
    - event_id, sport_key, commence_time
    - home_team, away_team
    - bookmaker_key, bookmaker_title, last_update
    - market_key (e.g., h2h), outcome_name, outcome_price
    - odds_snapshot_time (ingestion time)
    """
    if snapshot_time is None:
        snapshot_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
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
                            "odds_snapshot_time": snapshot_time,
                        }
                    )
    return rows


def best_price_by_outcome(
    rows: List[Dict[str, Any]], 
    mode: str = 'single_book'
) -> List[Dict[str, Any]]:
    """
    Select odds for events.
    
    Modes:
    - 'best_of': Max price per outcome across all books (Synthetic, Risk of unmatchable odds).
    - 'single_book': Pick ONE bookmaker per event (Best integrity).
    
    For 'single_book', we currently pick the bookmaker with the highest 'Home' odds 
    (or first available if not found) and use their entire H/D/A line.
    """
    if not rows:
        return []
        
    if mode == 'best_of':
        # Original logic: Max price per outcome
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
        
    else: # single_book (Default)
        # Group by event
        events: Dict[str, List[Dict]] = {}
        for r in rows:
            eid = r["event_id"]
            if eid not in events:
                events[eid] = []
            events[eid].append(r)
            
        final_rows = []
        
        for eid, event_rows in events.items():
            # Group by bookmaker
            books: Dict[str, List[Dict]] = {}
            for r in event_rows:
                bk = r["bookmaker_key"]
                if bk not in books:
                    books[bk] = []
                books[bk].append(r)
            
            # Pick best book
            # Criteria: Book with max price on 'Home' outcome?
            # Or just 'pinnacle' if exists?
            # Let's try: Find book with max Home odds.
            
            best_bk_key = None
            max_home_price = -1.0
            
            # First pass: Find best book
            for bk, items in books.items():
                # Find home price
                home_price = 0.0
                for item in items:
                    # Heuristic for home/away names if standardized, 
                    # but outcome_name is usually Team Name.
                    # We can use the fact that typically outcomes are sorted or we check vs home_team
                    if item.get("outcome_name") == item.get("home_team"):
                        try:
                            home_price = float(item["outcome_price"])
                        except:
                            pass
                        break
                
                if home_price > max_home_price:
                    max_home_price = home_price
                    best_bk_key = bk
            
            # Fallback: if no home match (e.g. draw/names mismatch), pick first book
            if not best_bk_key and books:
                best_bk_key = list(books.keys())[0]
                
            if best_bk_key:
                # Add all outcomes from this book
                final_rows.extend(books[best_bk_key])
                
        return final_rows
