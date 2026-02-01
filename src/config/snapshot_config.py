"""
Snapshot ML Configuration

Defines decision horizons and feature contract for snapshot-based training.
"""

from typing import Dict, List, Any

# Decision horizons (hours before kickoff)
# For now, we only have closing odds, so effectively T-0h
# Infrastructure supports multiple horizons for future use
DECISION_HORIZONS = [24, 6, 1, 0]  # 0 = closing (current data)

# Default horizon for training with current data
DEFAULT_HORIZON = 0  # Closing odds

# Feature configuration per horizon
FEATURE_WINDOWS = {
    24: {"form_games": 5, "min_rest_days": 0},
    6:  {"form_games": 5, "min_rest_days": 0},
    1:  {"form_games": 5, "min_rest_days": 0},
    0:  {"form_games": 5, "min_rest_days": 0},  # Closing odds
}

# Snapshot collection settings
SNAPSHOT_COLLECTION = {
    "interval_minutes": 30,
    "min_hours_before_kickoff": 0.5,  # Stop 30 min before
    "max_hours_before_kickoff": 168,  # Start 7 days before
}

# Feature contract - EXACT columns for train AND inference
FEATURE_COLUMNS: Dict[str, type] = {
    # Market features (from snapshot/odds)
    "odds_home": float,
    "odds_draw": float,
    "odds_away": float,
    "implied_home": float,
    "implied_draw": float,
    "implied_away": float,
    "no_vig_home": float,
    "no_vig_draw": float,
    "no_vig_away": float,
    "market_overround": float,
    "line_dispersion_home": float,
    "line_dispersion_draw": float,
    "line_dispersion_away": float,
    "book_count": int,
    
    # Football features (time-safe)
    "elo_home": float,
    "elo_away": float,
    "elo_diff": float,
    "form_pts_home_l5": float,
    "form_pts_away_l5": float,
    "form_gf_home_l5": float,
    "form_gf_away_l5": float,
    "form_ga_home_l5": float,
    "form_ga_away_l5": float,
    "rest_days_home": float,
    "rest_days_away": float,
    
    # Categorical (will be encoded by CatBoost)
    "league": str,
    "home_team": str,
    "away_team": str,
}

# Ordered list for consistent column ordering
FEATURE_ORDER: List[str] = list(FEATURE_COLUMNS.keys())

# Categorical feature names (for CatBoost)
CATEGORICAL_FEATURES: List[str] = ["league", "home_team", "away_team"]

# Numeric feature names
NUMERIC_FEATURES: List[str] = [f for f in FEATURE_ORDER if f not in CATEGORICAL_FEATURES]

def get_feature_contract() -> Dict[str, Any]:
    """Return the feature contract for validation."""
    return {
        "columns": FEATURE_ORDER,
        "types": FEATURE_COLUMNS,
        "categorical": CATEGORICAL_FEATURES,
        "numeric": NUMERIC_FEATURES,
        "version": "1.0.0",
    }
