from __future__ import annotations

from typing import Dict, List, Tuple


def implied_prob_from_decimal(odds_decimal: float) -> float:
    if odds_decimal <= 0:
        raise ValueError("odds_decimal must be > 0")
    return 1.0 / odds_decimal


def no_vig_proportional(implied_probs: Dict[str, float]) -> Dict[str, float]:
    """Remove bookmaker margin by proportional normalization.

    p_i = q_i / sum(q)
    where q are implied probabilities.
    This is NOT Shin, but it's stable and good as a baseline.
    """
    s = sum(implied_probs.values())
    if s <= 0:
        raise ValueError("sum implied probs must be > 0")
    return {k: v / s for k, v in implied_probs.items()}
