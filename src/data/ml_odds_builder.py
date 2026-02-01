"""
ML Odds Line Builder (Task B)

Constructs a clean, unbiased odds line for ML training and inference.
Separate from bet execution logic which uses best-of pricing.

Strategy:
1. Pinnacle-first: If Pinnacle present, use their line (most efficient market)
2. Fallback: Median consensus across all bookmakers

Also computes diagnostics for feature engineering:
- book_count: Number of bookmakers offering odds
- overround: Market overround (vig)
- dispersion_home/draw/away: Standard deviation across books
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Valid draw outcome names (case-insensitive)
DRAW_NAMES = {"draw", "tie", "x"}


@dataclass
class MLOddsLine:
    """Represents a validated 1X2 odds line for ML."""
    home_odds: float
    draw_odds: float
    away_odds: float
    
    # Market metadata
    source: str  # "pinnacle" or "median"
    book_count: int
    overround: float
    
    # Dispersion (std across books)
    dispersion_home: float
    dispersion_draw: float
    dispersion_away: float
    
    # No-vig probabilities
    no_vig_home: float
    no_vig_draw: float
    no_vig_away: float
    
    # Implied probabilities (with vig)
    implied_home: float
    implied_draw: float
    implied_away: float
    
    def is_valid(self) -> bool:
        """Check if all required fields are valid."""
        return (
            self.home_odds > 1.0 and
            self.draw_odds > 1.0 and
            self.away_odds > 1.0 and
            self.no_vig_home > 0.0 and
            self.no_vig_draw > 0.0 and
            self.no_vig_away > 0.0
        )


def normalize_outcome_name(name: str) -> str:
    """Normalize outcome name to lowercase, stripped."""
    return name.lower().strip()


def is_draw_outcome(outcome_name: str) -> bool:
    """
    Check if outcome name represents a draw.
    Recognizes: "Draw", "Tie", "X" (case-insensitive)
    """
    return normalize_outcome_name(outcome_name) in DRAW_NAMES


def classify_outcome(
    outcome_name: str,
    home_team: str,
    away_team: str
) -> Optional[str]:
    """
    Classify an outcome as 'H', 'D', or 'A'.
    
    Returns:
        'H' for home win
        'D' for draw
        'A' for away win
        None if cannot classify
    """
    norm_name = normalize_outcome_name(outcome_name)
    norm_home = normalize_outcome_name(home_team)
    norm_away = normalize_outcome_name(away_team)
    
    # Check draw first
    if norm_name in DRAW_NAMES:
        return 'D'
    
    # Check home team
    if norm_name == norm_home:
        return 'H'
    
    # Check away team
    if norm_name == norm_away:
        return 'A'
    
    # Try partial matching for common variations
    # e.g., "Man Utd" vs "Manchester United"
    if _fuzzy_team_match(norm_name, norm_home):
        return 'H'
    if _fuzzy_team_match(norm_name, norm_away):
        return 'A'
    
    return None


def _fuzzy_team_match(name1: str, name2: str) -> bool:
    """
    Fuzzy matching for team names.
    Handles abbreviations and common variations.
    """
    # Exact match
    if name1 == name2:
        return True
    
    # One contains the other
    if name1 in name2 or name2 in name1:
        return True
    
    # Word overlap (at least 2 common words)
    words1 = set(name1.split())
    words2 = set(name2.split())
    common = words1 & words2
    
    # Remove common generic words
    generic = {'fc', 'cf', 'afc', 'united', 'city', 'sporting', 'real', 'athletic'}
    meaningful_common = common - generic
    
    if len(meaningful_common) >= 1 and len(common) >= 2:
        return True
    
    return False


def extract_hda_odds_from_bookmaker(
    bookmaker_data: Dict[str, Any],
    home_team: str,
    away_team: str,
    market_key: str = "h2h"
) -> Optional[Tuple[float, float, float]]:
    """
    Extract H/D/A odds from a single bookmaker's data.
    
    Returns:
        Tuple (home_odds, draw_odds, away_odds) or None if invalid/incomplete
    """
    markets = bookmaker_data.get("markets", [])
    
    # Find h2h market
    h2h_market = None
    for m in markets:
        if m.get("key") == market_key:
            h2h_market = m
            break
    
    if not h2h_market:
        return None
    
    outcomes = h2h_market.get("outcomes", [])
    
    home_odds = None
    draw_odds = None
    away_odds = None
    
    for outcome in outcomes:
        name = outcome.get("name", "")
        price = outcome.get("price")
        
        if price is None or price <= 1.0:
            continue
        
        classification = classify_outcome(name, home_team, away_team)
        
        if classification == 'H':
            home_odds = float(price)
        elif classification == 'D':
            draw_odds = float(price)
        elif classification == 'A':
            away_odds = float(price)
    
    # Require all three outcomes
    if home_odds is None or draw_odds is None or away_odds is None:
        return None
    
    return (home_odds, draw_odds, away_odds)


def build_ml_odds_line(
    event: Dict[str, Any],
    market_key: str = "h2h"
) -> Optional[MLOddsLine]:
    """
    Build an ML odds line from raw event data.
    
    Strategy:
    1. Try Pinnacle first (if available)
    2. Fall back to median across all books
    
    Args:
        event: Raw event dict from Odds API with bookmakers
        market_key: Market to extract (default: h2h for 1X2)
    
    Returns:
        MLOddsLine if valid, None if event should be skipped
    """
    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")
    bookmakers = event.get("bookmakers", [])
    
    if not home_team or not away_team or not bookmakers:
        logger.debug(f"Skipping event: missing team names or bookmakers")
        return None
    
    # Extract all valid H/D/A lines
    all_lines: Dict[str, Tuple[float, float, float]] = {}
    
    for bk in bookmakers:
        bk_key = bk.get("key", "unknown")
        line = extract_hda_odds_from_bookmaker(bk, home_team, away_team, market_key)
        if line:
            all_lines[bk_key] = line
    
    if not all_lines:
        logger.debug(f"Skipping event {home_team} vs {away_team}: no valid H/D/A lines")
        return None
    
    # Strategy 1: Pinnacle first
    source = "median"
    if "pinnacle" in all_lines:
        chosen_line = all_lines["pinnacle"]
        source = "pinnacle"
    else:
        # Strategy 2: Median consensus
        home_odds_list = [line[0] for line in all_lines.values()]
        draw_odds_list = [line[1] for line in all_lines.values()]
        away_odds_list = [line[2] for line in all_lines.values()]
        
        chosen_line = (
            float(np.median(home_odds_list)),
            float(np.median(draw_odds_list)),
            float(np.median(away_odds_list))
        )
    
    home_odds, draw_odds, away_odds = chosen_line
    
    # Compute diagnostics
    book_count = len(all_lines)
    
    home_odds_list = [line[0] for line in all_lines.values()]
    draw_odds_list = [line[1] for line in all_lines.values()]
    away_odds_list = [line[2] for line in all_lines.values()]
    
    dispersion_home = float(np.std(home_odds_list)) if len(home_odds_list) > 1 else 0.0
    dispersion_draw = float(np.std(draw_odds_list)) if len(draw_odds_list) > 1 else 0.0
    dispersion_away = float(np.std(away_odds_list)) if len(away_odds_list) > 1 else 0.0
    
    # Implied probabilities (with vig)
    implied_home = 1.0 / home_odds
    implied_draw = 1.0 / draw_odds
    implied_away = 1.0 / away_odds
    
    # Overround
    overround = implied_home + implied_draw + implied_away
    
    # No-vig probabilities (normalized)
    no_vig_home = implied_home / overround
    no_vig_draw = implied_draw / overround
    no_vig_away = implied_away / overround
    
    return MLOddsLine(
        home_odds=home_odds,
        draw_odds=draw_odds,
        away_odds=away_odds,
        source=source,
        book_count=book_count,
        overround=overround,
        dispersion_home=dispersion_home,
        dispersion_draw=dispersion_draw,
        dispersion_away=dispersion_away,
        no_vig_home=no_vig_home,
        no_vig_draw=no_vig_draw,
        no_vig_away=no_vig_away,
        implied_home=implied_home,
        implied_draw=implied_draw,
        implied_away=implied_away
    )


def build_ml_odds_line_from_normalized_rows(
    rows: List[Dict[str, Any]],
    home_team: str,
    away_team: str
) -> Optional[MLOddsLine]:
    """
    Build ML odds line from already-normalized rows (flat format).
    
    Args:
        rows: List of normalized outcome rows for a single event
        home_team: Home team name
        away_team: Away team name
    
    Returns:
        MLOddsLine if valid, None if incomplete
    """
    # Group by bookmaker
    by_book: Dict[str, Dict[str, float]] = {}
    
    for row in rows:
        bk_key = row.get("bookmaker_key", "unknown")
        outcome_name = row.get("outcome_name", "")
        price = row.get("outcome_price")
        
        if price is None or float(price) <= 1.0:
            continue
        
        classification = classify_outcome(outcome_name, home_team, away_team)
        if classification is None:
            continue
        
        if bk_key not in by_book:
            by_book[bk_key] = {}
        
        by_book[bk_key][classification] = float(price)
    
    # Filter to books with complete H/D/A
    complete_books: Dict[str, Tuple[float, float, float]] = {}
    for bk_key, outcomes in by_book.items():
        if 'H' in outcomes and 'D' in outcomes and 'A' in outcomes:
            complete_books[bk_key] = (outcomes['H'], outcomes['D'], outcomes['A'])
    
    if not complete_books:
        return None
    
    # Pinnacle first
    source = "median"
    if "pinnacle" in complete_books:
        chosen_line = complete_books["pinnacle"]
        source = "pinnacle"
    else:
        home_list = [line[0] for line in complete_books.values()]
        draw_list = [line[1] for line in complete_books.values()]
        away_list = [line[2] for line in complete_books.values()]
        
        chosen_line = (
            float(np.median(home_list)),
            float(np.median(draw_list)),
            float(np.median(away_list))
        )
    
    home_odds, draw_odds, away_odds = chosen_line
    book_count = len(complete_books)
    
    # Dispersion
    home_list = [line[0] for line in complete_books.values()]
    draw_list = [line[1] for line in complete_books.values()]
    away_list = [line[2] for line in complete_books.values()]
    
    dispersion_home = float(np.std(home_list)) if len(home_list) > 1 else 0.0
    dispersion_draw = float(np.std(draw_list)) if len(draw_list) > 1 else 0.0
    dispersion_away = float(np.std(away_list)) if len(away_list) > 1 else 0.0
    
    # Implied and no-vig
    implied_home = 1.0 / home_odds
    implied_draw = 1.0 / draw_odds
    implied_away = 1.0 / away_odds
    overround = implied_home + implied_draw + implied_away
    
    return MLOddsLine(
        home_odds=home_odds,
        draw_odds=draw_odds,
        away_odds=away_odds,
        source=source,
        book_count=book_count,
        overround=overround,
        dispersion_home=dispersion_home,
        dispersion_draw=dispersion_draw,
        dispersion_away=dispersion_away,
        no_vig_home=implied_home / overround,
        no_vig_draw=implied_draw / overround,
        no_vig_away=implied_away / overround,
        implied_home=implied_home,
        implied_draw=implied_draw,
        implied_away=implied_away
    )
