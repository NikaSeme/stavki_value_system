from __future__ import annotations

from typing import Dict, List, Any
from .no_vig import implied_prob_from_decimal, no_vig_proportional


def compute_ev(p_true: float, odds_decimal: float) -> float:
    """Expected value per 1 unit stake: p*odds - 1"""
    return p_true * odds_decimal - 1.0


def compute_value_bets_1x2(
    best_rows: List[Dict[str, Any]],
    model_probs: Dict[str, Dict[str, float]],
    *,
    market_key: str = "h2h",
    use_no_vig: bool = True,
    ev_threshold: float = 0.05,
) -> List[Dict[str, Any]]:
    """Compute value bets for 1X2 (or h2h) using best odds across bookmakers.

    model_probs[event_id] = {"home": p, "away": p, "draw": p(optional)}
    We map outcome_name by exact team names. For soccer 1X2, the Odds API usually returns:
    - home team name
    - away team name
    - Draw (if market supports draw)
    """
    # group best_rows by event
    by_event: Dict[str, List[Dict[str, Any]]] = {}
    for r in best_rows:
        if r.get("market_key") != market_key:
            continue
        by_event.setdefault(r["event_id"], []).append(r)

    value_bets: List[Dict[str, Any]] = []
    for event_id, rows in by_event.items():
        mp = model_probs.get(event_id)
        if not mp:
            continue

        # build implied probs from best odds
        implied: Dict[str, float] = {}
        odds: Dict[str, float] = {}
        for r in rows:
            name = r["outcome_name"]
            o = float(r["outcome_price"])
            odds[name] = o
            implied[name] = implied_prob_from_decimal(o)

        fair = no_vig_proportional(implied) if use_no_vig else implied

        # map model probs to the same names
        for r in rows:
            name = r["outcome_name"]
            o = float(r["outcome_price"])

            # try keys: exact name, or special "Draw"
            if name in mp:
                p_true = float(mp[name])
            elif name.lower() == "draw" and ("draw" in mp):
                p_true = float(mp["draw"])
            else:
                # can't align
                continue

            ev = compute_ev(p_true, o)
            if ev >= ev_threshold:
                value_bets.append(
                    {
                        "event_id": event_id,
                        "market": market_key,
                        "selection": name,
                        "odds": o,
                        "bookmaker": r.get("bookmaker_title"),
                        "p_model": p_true,
                        "p_implied_novig": float(fair.get(name, implied.get(name, 0.0))),
                        "ev": ev,
                    }
                )
    # sort by EV descending
    value_bets.sort(key=lambda x: x["ev"], reverse=True)
    return value_bets
