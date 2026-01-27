"""
Live value bet finder module.

Loads latest odds, computes EV using model probabilities, and ranks value opportunities.
"""

from __future__ import annotations

import csv
import glob
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ... (imports continue)

# ... (function defs)




from .no_vig import implied_prob_from_decimal, no_vig_proportional
from .value import compute_ev
from .staking import fractional_kelly

# Team name normalization mapping (common aliases)
TEAM_ALIASES = {
    'wolves': 'wolverhampton wanderers',
    'man city': 'manchester city',
    'man utd': 'manchester united',
    'man united': 'manchester united',
    'newcastle': 'newcastle united',
    'west ham': 'west ham united',
    'brighton': 'brighton and hove albion',
    'tottenham': 'tottenham hotspur',
    'spurs': 'tottenham hotspur',
    'nottm forest': 'nottingham forest',
    'notts forest': 'nottingham forest',
}


def normalize_team_name(team_name: str) -> str:
    """
    Normalize team name for consistent matching.
    
    Args:
        team_name: Raw team name from odds data
        
    Returns:
        Normalized team name (lowercase, no punctuation, aliases resolved)
    """
    import re
    
    # Lowercase and strip
    name = team_name.lower().strip()
    
    # Remove punctuation except spaces
    name = re.sub(r'[^a-z0-9\s]', '', name)
    
    # Collapse multiple spaces
    name = ' '.join(name.split())
    
    # Check for aliases
    if name in TEAM_ALIASES:
        name = TEAM_ALIASES[name]
    
    return name


def validate_outcome_mapping(
    outcome_name: str,
    home_team: str,
    away_team: str,
    model_probs: Dict[str, float]
) -> Optional[float]:
    """
    Validate and map outcome name to model probability.
    
    Args:
        outcome_name: Outcome from odds (e.g., team name or 'Draw')
        home_team: Home team name
        away_team: Away team name
        model_probs: Model probabilities dict
        
    Returns:
        Model probability if valid mapping found, None otherwise
    """
    # Normalize all names
    norm_outcome = normalize_team_name(outcome_name)
    norm_home = normalize_team_name(home_team)
    norm_away = normalize_team_name(away_team)
    
    # Check for draw
    if outcome_name.lower().strip() == 'draw':
        return model_probs.get('Draw')
    
    # Check if outcome matches home or away
    if norm_outcome == norm_home:
        # Look for probability under original home_team name
        return model_probs.get(home_team)
    elif norm_outcome == norm_away:
        # Look for probability under original away_team name
        return model_probs.get(away_team)
    
    # No valid mapping
    return None


def validate_prob_sum(
    probs: Dict[str, float],
    tolerance: float = 0.02
) -> bool:
    """
    Check if probabilities sum to 1.0 within tolerance.
    
    Args:
        probs: Probability dictionary
        tolerance: Allowed deviation from 1.0
        
    Returns:
        True if sum is valid
    """
    total = sum(probs.values())
    return abs(total - 1.0) <= tolerance


def renormalize_probs(probs: Dict[str, float]) -> Dict[str, float]:
    """
    Renormalize probabilities to sum to 1.0.
    
    Args:
        probs: Probability dictionary
        
    Returns:
        Normalized probabilities
    """
    total = sum(probs.values())
    if total <= 0:
        return probs
    return {k: v / total for k, v in probs.items()}


def check_outlier_odds(
    all_odds: List[float],
    gap_threshold: float = 0.20
) -> bool:
    """
    Check if best odds is an outlier (>gap_threshold above second-best).
    
    Args:
        all_odds: List of odds for this outcome from different bookmakers
        gap_threshold: Max allowed gap (default 20%)
        
    Returns:
        True if best odds is an outlier
    """
    if len(all_odds) < 2:
        return False
    
    sorted_odds = sorted(all_odds, reverse=True)
    best = sorted_odds[0]
    second = sorted_odds[1]
    
    gap = (best - second) / second
    return gap > gap_threshold


def check_high_odds_confirmation(
    all_odds: List[float],
    odds_threshold: float = 10.0,
    similarity_threshold: float = 0.10
) -> bool:
    """
    Check if high odds has confirmation from multiple bookmakers.
    
    Args:
        all_odds: List of odds for this outcome
        odds_threshold: Threshold to consider "high odds"
        similarity_threshold: Max difference for "similar" odds (10%)
        
    Returns:
        True if confirmed by >=2 bookmakers
    """
    if max(all_odds) < odds_threshold:
        return True  # Not high odds, no confirmation needed
    
    best = max(all_odds)
    
    # Count how many bookmakers have similar odds
    similar_count = sum(
        1 for o in all_odds
        if abs(o - best) / best <= similarity_threshold
    )
    
    return similar_count >= 2


def load_latest_odds(sport: str, odds_dir: str = "outputs/odds") -> Optional[pd.DataFrame]:
    """
    Load the most recent normalized odds CSV for the given sport.
    
    Args:
        sport: Sport key (e.g., 'soccer_epl')
        odds_dir: Directory containing odds files
        
    Returns:
        DataFrame with latest odds, or None if no files found
    """
    pattern = os.path.join(odds_dir, f"normalized_{sport}_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Sort by filename (contains timestamp) to get latest
    latest_file = sorted(files)[-1]
    
    try:
        df = pd.read_csv(latest_file)
        df['_source_file'] = latest_file
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load {latest_file}: {e}")


def select_best_prices(
    df: pd.DataFrame,
    check_outliers: bool = False,
    outlier_gap: float = 0.20
) -> pd.DataFrame:
    """
    For each event+market+outcome, select the bookmaker offering the best (highest) price.
    
    Args:
        df: DataFrame with columns: event_id, market_key, outcome_name, outcome_price, bookmaker_title
        check_outliers: If True, exclude outlier odds
        outlier_gap: Gap threshold for outlier detection
        
    Returns:
        DataFrame with best prices only
    """
    if not check_outliers:
        # Simple: just pick max
        idx = df.groupby(['event_id', 'market_key', 'outcome_name'])['outcome_price'].idxmax()
        return df.loc[idx].reset_index(drop=True)
    
    # Check for outliers and exclude them
    result_rows = []
    
    for (event_id, market_key, outcome_name), group in df.groupby(
        ['event_id', 'market_key', 'outcome_name']
    ):
        all_odds = group['outcome_price'].tolist()
        
        # Check if best is an outlier
        if check_outlier_odds(all_odds, outlier_gap):
            # Use second-best instead
            sorted_indices = group['outcome_price'].argsort().iloc[::-1]
            if len(sorted_indices) > 1:
                best_idx = group.iloc[sorted_indices[1]].name
            else:
                # Only one bookmaker, skip
                continue
        else:
            # Use best
            best_idx = group['outcome_price'].idxmax()
        
        result_rows.append(df.loc[best_idx])
    
    if not result_rows:
        return pd.DataFrame()
    
    return pd.DataFrame(result_rows).reset_index(drop=True)


def compute_no_vig_probs(best_prices: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute no-vig probabilities for each event.
    
    Args:
        best_prices: DataFrame with best odds per outcome
        
    Returns:
        Dict[event_id][outcome_name] = no-vig probability
    """
    result = {}
    
    # Group by event and market
    for (event_id, market_key), group in best_prices.groupby(['event_id', 'market_key']):
        implied = {}
        for _, row in group.iterrows():
            outcome = row['outcome_name']
            odds = float(row['outcome_price'])
            implied[outcome] = implied_prob_from_decimal(odds)
        
        # Apply no-vig normalization
        try:
            no_vig = no_vig_proportional(implied)
            
            # Store with event_id as key
            if event_id not in result:
                result[event_id] = {}
            result[event_id].update(no_vig)
        except ValueError:
            # Skip if invalid odds
            continue
    
    return result


# Global ML model loader (initialized once at startup)
_ensemble = None
_ml_feature_extractor = None


def initialize_ml_model():
    """
    Initialize ML model at startup.
    
    Loads Ensemble (Poisson + CatBoost + Neural) and Extractor.
    Raises RuntimeError if loading fails.
    """
    global _ensemble, _ml_feature_extractor
    
    from src.models.ensemble_predictor import EnsemblePredictor
    from src.features.live_extractor import LiveFeatureExtractor
    
    if _ensemble is None:
        try:
            # Initialize with Neural Model enabled (Grand Unification)
            _ensemble = EnsemblePredictor(use_neural=True)
            # logger.info("✓ Ensemble initialized") # initialized in class
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Ensemble model: {e}\n"
                "Run scripts/train_model.py first."
            )
    
    if _ml_feature_extractor is None:
        # Load state_file for Elo persistence
        state_file = Path('data/live_extractor_state.pkl')
        _ml_feature_extractor = LiveFeatureExtractor(state_file=state_file)


def get_model_probabilities(
    events: pd.DataFrame,
    odds_df: Optional[pd.DataFrame] = None,
    model_type: str = "ml"
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict]]:
    """
    Get model-based probabilities for each event using Ensemble.
    
    Args:
        events: DataFrame with event details (home_team, away_team, etc.)
        odds_df: Current odds data (required for ML model)
        model_type: 'ml' (Ensemble), 'simple' (baseline), or 'market'
        
    Returns:
        tuple (probs_dict, sentiment_data)
        - probs_dict: Dict[event_id][outcome_name] = model probability
        - sentiment_data: Dict[event_id] = {home: {...}, away: {...}}
    """
    if model_type == "ml":
        # SAFETY CHECK: Only run ML on Soccer (Training Domain)
        # Models (A/B/C) were trained on Soccer data structure.
        if not events.empty:
            sport = events.iloc[0].get('sport_key', '')
            if 'soccer' not in sport:
                logger.warning(f"ML Ensemble trained on Soccer only. Skipping {sport} (Falling back to Simple).")
                return _get_simple_model_probs(events), {}

        # Ensemble mode
        if _ensemble is None:
            raise RuntimeError(
                "Ensemble model not initialized. Call initialize_ml_model() first."
            )
        
        if odds_df is None:
            raise ValueError("odds_df required for ML model")
        
        # 1. Extract features using persistent extractor (preserves Elo)
        X = _ml_feature_extractor.extract_features(events, odds_df)
        
        # Guardrail: Sanitize Numeric Features without destroying Categorical ones
        numeric_cols = [c for c in X.columns if c not in ['HomeTeam', 'AwayTeam', 'Season']]
        X[numeric_cols] = X[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        
        # 2. Fetch Sentiment (Model C)
        sentiment_data = _ml_feature_extractor.fetch_sentiment_for_events(events)
        
        # 3. Dynamic Feature Alignment (M40) - Target CatBoost requirements
        # Ensemble.catboost is the ModelLoader instance
        if _ensemble.catboost:
            expected_cols = _ensemble.catboost.get_feature_names()
            
            if expected_cols:
                # Add missing columns (fill with 0.0)
                missing_cols = set(expected_cols) - set(X.columns)
                if missing_cols:
                    for col in missing_cols:
                        X[col] = 0.0
                
                # Reorder and Select (drops extra columns naturally used by Neural?)
                # WAIT: Neural needs 22 features. CatBoost needs 10.
                # If we filter X here for CatBoost, Neural might fail if it expects 22.
                # Ensemble.predict handles this?
                # Neural expects 22. CatBoost matches columns by name (if dataframe) or order?
                # ModelLoader.predict expects numpy array matching scaler.
                # If Neural needs ALL features, we shouldn't drop them here.
                # We should ensure X has at least the UNION of needed features?
                # Actually, Neural v1 was trained on 'epl_features_2021_2024.csv' (22 cols).
                # CatBoost v1 was trained on 10 cols.
                # If we modify X to match CatBoost, Neural breaks.
                
                # FIX: Pass the FULL X to Ensemble. 
                # Ensemble needs to handle column selection for CatBoost internally?
                # No, Ensemble calls `self.catboost.predict(X.values)`.
                # If X has 22 cols, and CatBoost scaler expects 10, it CRASHES.
                
                # We need to create X_catboost (10 cols) and X_neural (22 cols).
                # But Ensemble logic is: `catboost.predict(X.values)`.
                
                # I must update EnsemblePredictor to handle this split if I want it to work.
                # Currently EnsemblePredictor just passes X.values to both.
                # If their shapes differ, one will crash.
                pass 

        # CRITICAL FIX for Feature Mismatch:
        # Neural needs 22 features (from extract_features).
        # CatBoost needs 10 features (subset).
        # I cannot filter X here without breaking Neural.
        # I rely on EnsemblePredictor to handle this? 
        # Checking EnsemblePredictor again... it passes `X.values`.
        # This will CRASH CatBoost if X has 22 cols.
        # I MUST FIX EnsemblePredictor to filter columns for CatBoost.
        
        # For now, I will assume Ensemble logic needs fixing. 
        # But I am editing value_live.py.
        # I will pass the FULL X to ensemble, and trust I fix ensemble in next step?
        # Or I do it here? No, 'predict' interface is clean.
        
        # Let's proceed with passing X. I will subsequently patch EnsemblePredictor to be smart about features.
        
        # 4. Predict
        # Pass full X (features)
        probs_array, components = _ensemble.predict(
            events, odds_df, sentiment_data=sentiment_data, features=X
        )
        
        # Convert to dict format
        probs_dict = {}
        for i, (idx, event) in enumerate(events.iterrows()):
            event_id = event['event_id']
            
            # Extract probabilities for this event
            p_home = float(probs_array[i, 0])
            p_draw = float(probs_array[i, 1])
            p_away = float(probs_array[i, 2])
            
            probs_dict[event_id] = {
                event['home_team']: p_home,
                'Draw': p_draw,
                event['away_team']: p_away,
            }
            
        return probs_dict, sentiment_data
        
    elif model_type == "simple":
        # Baseline model (only for debugging)
        return _get_simple_model_probs(events), {}
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _get_simple_model_probs(
    events: pd.DataFrame,
    market_key: str = 'h2h'
) -> Dict[str, Dict[str, float]]:
    """
    Simple baseline model using symmetric probabilities with slight home advantage.
    
    This is a placeholder. In production, use:
    - ELO ratings lookup
    - Poisson model with team strength parameters
    - Historical head-to-head data
    """
    probs = {}
    
    for _, event in events.iterrows():
        event_id = event['event_id']
        home_team = event['home_team']
        away_team = event['away_team']
        
        # Simple baseline: 45% home, 30% away for 2-way markets
        if market_key == 'h2h' and event.get('sport_key', '').startswith('soccer'):
            # 3-way market (home/draw/away)
            probs[event_id] = {
                home_team: 0.40,
                away_team: 0.30,
                'Draw': 0.30,
            }
        else:
            # 2-way market
            probs[event_id] = {
                home_team: 0.55,
                away_team: 0.45,
            }
    
    return probs


def check_model_market_divergence(
    p_model: float,
    p_market: float,
    threshold: float = 0.20
) -> tuple[bool, float, str]:
    """
    Check if model probability diverges significantly from market.
    
    Args:
        p_model: Model probability
        p_market: Market (no-vig) probability
        threshold: Divergence threshold (default: 0.20 = 20 percentage points)
        
    Returns:
        (is_safe, divergence, level)
        - is_safe: True if divergence <= threshold
        - divergence: Absolute difference (p_model - p_market)
        - level: "safe", "caution", "extreme"
    """
    divergence = p_model - p_market
    abs_div = abs(divergence)
    
    if abs_div <= threshold:
        level = "safe"
        is_safe = True
    elif abs_div <= threshold * 2:  # 20-40%
        level = "caution"
        is_safe = False
    else:  # >40%
        level = "extreme"
        is_safe = False
    
    return is_safe, divergence, level


def is_baseline_model_output(
    model_probs: Dict[str, Dict[str, float]],
    sample_size: int = 3
) -> bool:
    """
    Detect if probabilities match baseline model pattern.
    
    Baseline model uses fixed probabilities:
    - Soccer 3-way: 0.40 home, 0.30 away, 0.30 draw
    - Other 2-way: 0.55 home, 0.45 away
    
    Args:
        model_probs: Model probabilities for all events
        sample_size: Number of events to check
        
    Returns:
        True if baseline pattern detected
    """
    if not model_probs or len(model_probs) == 0:
        return False
    
    # Check first few events
    samples = list(model_probs.values())[:sample_size]
    
    baseline_3way_pattern = {0.40, 0.30, 0.30}
    baseline_2way_pattern = {0.55, 0.45}
    
    matches = 0
    for probs in samples:
        prob_values = set(round(p, 2) for p in probs.values())
        
        if prob_values == baseline_3way_pattern or prob_values == baseline_2way_pattern:
            matches += 1
    
    # If >50% of samples match baseline pattern, it's likely baseline
    return matches > len(samples) / 2


def get_sentiment_multiplier(sentiment_features: Dict[str, Any]) -> float:
    """
    Calculate a multiplier based on team sentiment features.
    
    Logic:
    - Neutral/No data: 1.0x
    - Highly Positive (> 0.5): 1.1x to 1.2x
    - Negative (< -0.2): 0.5x (Safety first)
    - Toxic (< -0.5): 0.2x (Extreme De-risking)
    
    Args:
        sentiment_features: Dict containing 'compound', 'magnitude', etc.
        
    Returns:
        Multiplier (0.2 to 1.2)
    """
    if not sentiment_features or 'compound' not in sentiment_features:
        return 1.0
        
    score = sentiment_features.get('compound', 0.0)
    
    if score < -0.5:
        return 0.2  # Toxic Trap
    elif score < -0.2:
        return 0.5  # Significant negativity
    elif score > 0.5:
        return 1.2  # Bullish momentum
    elif score > 0.2:
        return 1.1  # Positive vibe
        
    return 1.0


def calculate_justified_score(
    p_model: float,
    p_market: float,
    odds: float,
    ev: float,
    sentiment_multiplier: float = 1.0
) -> int:
    """
    Calculate a 0-100 score for how "justified" a value bet is.
    
    High scores = likely real value (model has edge)
    Low scores = likely model error (miscalibration)
    
    Factors:
    - Model-market divergence (larger = lower score)
    - Odds level (extreme odds = lower score)
    - EV magnitude (extreme EV = lower score unless well-supported)
    - Sentiment (toxic sentiment = major penalty)
    """
    score = 100
    
    # Penalty for model-market divergence
    divergence = abs(p_model - p_market)
    if divergence > 0.40:  # Extreme (>40pp)
        score -= 60
    elif divergence > 0.20:  # Large (20-40pp)
        score -= 40
    elif divergence > 0.10:  # Moderate (10-20pp)
        score -= 20
    elif divergence > 0.05:  # Small (5-10pp)
        score -= 10
    
    # Penalty for very high odds (harder to estimate accurately)
    if odds > 20:
        score -= 30
    elif odds > 10:
        score -= 20
    elif odds > 6:
        score -= 10
    
    # Penalty for extreme EV without justification
    if ev > 2.0 and divergence > 0.30:  # >200% EV with large divergence
        score -= 40
    elif ev > 1.0 and divergence > 0.20:  # >100% EV with moderate divergence
        score -= 20

    # Sentiment Penalty
    if sentiment_multiplier < 0.6:
        score -= 50  # Major trap warning
    elif sentiment_multiplier < 1.0:
        score -= 20
    
    return max(0, min(100, score))



def compute_ev_candidates(
    model_probs: Dict[str, Dict[str, float]],
    best_prices: pd.DataFrame,
    all_odds_df: pd.DataFrame = None,
    sentiment_data: Dict[str, Dict] = None,
    threshold: float = 0.05,
    market_key: str = "h2h",

    prob_sum_tol: float = 0.02,
    drop_bad_prob_sum: bool = True,
    renormalize:bool = False,
    confirm_high_odds: bool = False,
    high_odds_threshold: float = 10.0,
    high_odds_p_threshold: float = 0.15,
    cap_high_odds_prob: Optional[float] = None,
    alpha_shrink: float = 1.0,
    max_model_market_div: Optional[float] = None,
    drop_extreme_div: bool = False,
    bankroll: float = 1000.0,
) -> List[Dict[str, Any]]:
    """
    Compute expected value for all bets with validation and guardrails.
    
    Args:
        model_probs: Model probabilities per event and outcome
        best_prices: Best odds per outcome
        all_odds_df: Full odds data for confirmation checks (optional)
        threshold: Minimum EV to include
        market_key: Market to analyze (e.g., 'h2h')
        prob_sum_tol: Probability sum tolerance
        drop_bad_prob_sum: Drop events with bad probability sums
        renormalize: Renormalize probabilities instead of dropping
        confirm_high_odds: Require multi-bookmaker confirmation for high odds
        high_odds_threshold: Odds threshold
        high_odds_p_threshold: Min probability for high odds warning
        cap_high_odds_prob: Cap probability for high odds (None=no cap)
        alpha_shrink: Market shrinkage factor (1.0=no shrinkage)
        max_model_market_div: Max allowed model-market divergence (None=no check)
        drop_extreme_div: Drop bets with extreme divergence (>40%)
        bankroll: Total bankroll for stake sizing (default 1000.0)
        
    Returns:
        List of value bet candidates with EV, odds, probabilities, metadata
    """
    candidates = []
    
    # Compute no-vig probabilities for market shrinkage
    no_vig_probs = compute_no_vig_probs(best_prices)
    
    # Validate model probabilities
    validated_events = set()
    for event_id, probs in model_probs.items():
        # Check probability sum
        if not validate_prob_sum(probs, prob_sum_tol):
            if renormalize:
                model_probs[event_id] = renormalize_probs(probs)
            elif drop_bad_prob_sum:
                continue  # Skip this event
        validated_events.add(event_id)
    
    # Filter to specified market
    market_df = best_prices[best_prices['market_key'] == market_key]
    
    for _, row in market_df.iterrows():
        event_id = row['event_id']
        outcome = row['outcome_name']
        odds = float(row['outcome_price'])
        home_team = row.get('home_team', '')
        away_team = row.get('away_team', '')
        
        # Skip if event was invalidated
        if event_id not in validated_events:
            continue
        
        # Get model probability with validation
        event_probs = model_probs.get(event_id, {})
        p_model = validate_outcome_mapping(outcome, home_team, away_team, event_probs)
        
        if p_model is None:
            # Outcome mapping failed, skip
            continue
        
        # Check high odds confirmation if enabled
        if confirm_high_odds and odds >= high_odds_threshold and p_model >= high_odds_p_threshold:
            if all_odds_df is not None:
                # Get all odds for this event/outcome
                same_outcome = all_odds_df[
                    (all_odds_df['event_id'] == event_id) &
                    (all_odds_df['market_key'] == market_key) &
                    (all_odds_df['outcome_name'] == outcome)
                ]
                all_odds = same_outcome['outcome_price'].tolist()
                
                if not check_high_odds_confirmation(all_odds, high_odds_threshold):
                    # Not confirmed, skip
                    continue
        
        # Apply probability capping for high odds if specified
        p_final = p_model
        if cap_high_odds_prob is not None and odds > high_odds_threshold:
            p_final = min(p_model, cap_high_odds_prob)
        
        # Apply market shrinkage if specified
        if alpha_shrink < 1.0:
            p_novig = no_vig_probs.get(event_id, {}).get(outcome, p_model)
            p_final = alpha_shrink * p_final + (1 - alpha_shrink) * p_novig
        
        # Get market probability for metadata
        p_market = no_vig_probs.get(event_id, {}).get(outcome, implied_prob_from_decimal(odds))
        
        # Check model-market divergence if enabled
        if max_model_market_div is not None or drop_extreme_div:
            is_safe, divergence, level = check_model_market_divergence(
                p_model, p_market, max_model_market_div or 0.20
            )
            
            # Drop if extreme divergence and drop_extreme_div enabled
            if drop_extreme_div and level == "extreme":
                continue
            
            # Drop if above max_model_market_div threshold
            if max_model_market_div is not None and not is_safe:
                continue
        else:
            is_safe, divergence, level = check_model_market_divergence(p_model, p_market, 1.0)
        
        # Calculate EV
        ev = compute_ev(p_final, odds)
        
        # Phase 3: Decision Intelligence - Sentiment Quality
        # Get multiplier for the specific outcome (home or away)
        s_multiplier = 1.0
        if sentiment_data and event_id in sentiment_data:
            # Check if outcome is home or away to get correct team sentiment
            features = sentiment_data[event_id]
            if outcome == home_team:
                s_multiplier = get_sentiment_multiplier(features.get('home', {}))
            elif outcome == away_team:
                s_multiplier = get_sentiment_multiplier(features.get('away', {}))
        
        # Calculate Quality Score
        quality_score = ev * s_multiplier
        
        # Calculate justified score  
        justified_score = calculate_justified_score(
            p_model, p_market, odds, ev, sentiment_multiplier=s_multiplier
        )
        
        if ev >= threshold:
            # Calculate Stake (Phase 3: Stake Adjusted by Quality)
            # We shrink the probability directed proportionally by quality for staking
            # Toxic sentiment -> lower quality -> smaller stake
            p_staked = p_final * s_multiplier
            
            stake = fractional_kelly(
                probability=p_staked,
                odds=odds,
                bankroll=bankroll,
                fraction=0.5, # Half Kelly
                max_stake_pct=5.0 # Max 5%
            )
            
            candidates.append({
                'event_id': event_id,
                'sport_key': row.get('sport_key', ''),
                'commence_time': row.get('commence_time', ''),
                'home_team': home_team,
                'away_team': away_team,
                'market': market_key,
                'selection': outcome,
                'odds': odds,
                'bookmaker': row['bookmaker_title'],
                'bookmaker_key': row.get('bookmaker_key', ''),
                'p_model': round(p_model, 4),
                'p_final': round(p_final, 4) if p_final != p_model else round(p_model, 4),
                'p_implied': round(implied_prob_from_decimal(odds), 4),
                'p_market': round(p_market, 4),
                'model_market_div': round(divergence, 4),
                'divergence_level': level,
                'sentiment_multiplier': round(s_multiplier, 2),
                'quality_score': round(quality_score, 4),
                'justified_score': justified_score,
                'ev': round(ev, 4),
                'ev_pct': round(ev * 100, 2),
                'stake': round(stake, 2),
                'bankroll': bankroll,
                'stake_pct': round((stake/bankroll)*100, 2)
            })

    
    return candidates


def rank_value_bets(candidates: List[Dict[str, Any]], top_n: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Sort value bets by EV and optionally limit to top N.
    
    Args:
        candidates: List of value bet candidates
        top_n: Maximum number to return (None = all)
        
    Returns:
        Sorted list of top value bets
    """
    sorted_bets = sorted(candidates, key=lambda x: x['ev'], reverse=True)
    
    if top_n is not None:
        return sorted_bets[:top_n]
    
    return sorted_bets


def save_value_bets(
    bets: List[Dict[str, Any]],
    sport: str,
    output_dir: str = "outputs/value"
) -> Tuple[Path, Path]:
    """
    Save value bets to CSV and JSON files.
    
    Args:
        bets: List of value bet dictionaries
        sport: Sport key for filename
        output_dir: Output directory
        
    Returns:
        Tuple of (csv_path, json_path)
    """
    import json
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save CSV
    csv_file = output_path / f"value_{sport}_{timestamp}.csv"
    if bets:
        fieldnames = list(bets[0].keys())
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(bets)
    else:
        # Empty file with headers
        with open(csv_file, 'w') as f:
            f.write("# No value bets found\n")
    
    # Save JSON
    json_file = output_path / f"value_{sport}_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump({
            'sport': sport,
            'timestamp': timestamp,
            'count': len(bets),
            'bets': bets
        }, f, indent=2)
    
    return csv_file, json_file


def diagnose_ev_outliers(
    candidates: List[Dict[str, Any]],
    all_odds_df: pd.DataFrame,
    model_probs: Dict[str, Dict[str, float]],
    no_vig_probs: Dict[str, Dict[str, float]],
    top_k: int = 20,
    output_dir: str = "outputs/diagnostics"
) -> str:
    """
    Diagnose potential issues with top EV candidates.
    
    Args:
        candidates: Value bet candidates
        all_odds_df: Full odds DataFrame for analysis
        model_probs: Model probabilities
        no_vig_probs: No-vig market probabilities
        top_k: Number of top candidates to analyze
        output_dir: Output directory for report
        
    Returns:
        Path to diagnostics report file
    """
    from pathlib import Path
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get top K by EV
    top_bets = sorted(candidates, key=lambda x: x['ev'], reverse=True)[:top_k]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(output_dir) / f"ev_diagnostics_{timestamp}.txt"
    
    lines = []
    lines.append("=" * 80)
    lines.append("EV DIAGNOSTICS REPORT")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total candidates: {len(candidates)}")
    lines.append(f"Analyzing top: {len(top_bets)}")
    lines.append("")
    
    for i, bet in enumerate(top_bets, 1):
        lines.append("-" * 80)
        lines.append(f"#{i} - EV: +{bet['ev_pct']:.1f}%")
        lines.append("-" * 80)
        
        # Basic info
        lines.append(f"Event: {bet['home_team']} vs {bet['away_team']}")
        lines.append(f"Selection: {bet['selection']} @ {bet['odds']}")
        lines.append(f"Bookmaker: {bet['bookmaker']}")
        lines.append(f"Start time: {bet['commence_time']}")
        lines.append("")
        
        # Probabilities
        lines.append("Probabilities:")
        lines.append(f"  Model (p_model): {bet['p_model']:.1%}")
        if 'p_final' in bet and bet['p_final'] != bet['p_model']:
            lines.append(f"  Final (adjusted): {bet['p_final']:.1%}")
        lines.append(f"  Implied: {bet['p_implied']:.1%}")
        
        # Get no-vig probability
        event_id = bet['event_id']
        outcome = bet['selection']
        p_novig = no_vig_probs.get(event_id, {}).get(outcome)
        if p_novig:
            lines.append(f"  No-vig market: {p_novig:.1%}")
        lines.append("")
        
        # Outcome mapping check
        home = bet['home_team']
        away = bet['away_team']
        norm_outcome = normalize_team_name(outcome)
        norm_home = normalize_team_name(home)
        norm_away = normalize_team_name(away)
        
        lines.append("Outcome Mapping:")
        lines.append(f"  Normalized outcome: '{norm_outcome}'")
        lines.append(f"  Normalized home: '{norm_home}'")
        lines.append(f"  Normalized away: '{norm_away}'")
        
        if norm_outcome == norm_home:
            lines.append(f"  ✓ Outcome matches HOME team")
        elif norm_outcome == norm_away:
            lines.append(f"  ✓ Outcome matches AWAY team")
        elif outcome.lower().strip() == 'draw':
            lines.append(f"  ✓ Outcome is DRAW")
        else:
            lines.append(f"  ⚠️  WARNING: Outcome mapping unclear!")
        lines.append("")
        
        # Probability sum check
        event_probs = model_probs.get(event_id, {})
        if event_probs:
            prob_sum = sum(event_probs.values())
            lines.append(f"Probability Sum Check:")
            lines.append(f"  Sum: {prob_sum:.4f}")
            if abs(prob_sum - 1.0) <= 0.02:
                lines.append(f"  ✓ Valid (within 2% tolerance)")
            else:
                lines.append(f"  ⚠️  WARNING: Sum deviates from 1.0!")
        lines.append("")
        
        # Odds outlier check
        same_outcome = all_odds_df[
            (all_odds_df['event_id'] == event_id) &
            (all_odds_df['market_key'] == bet['market']) &
            (all_odds_df['outcome_name'] == outcome)
        ]
        
        if len(same_outcome) > 0:
            all_odds_for_outcome = same_outcome['outcome_price'].tolist()
            lines.append(f"Bookmaker Odds Coverage:")
            lines.append(f"  # Bookmakers: {len(all_odds_for_outcome)}")
            lines.append(f"  Min odds: {min(all_odds_for_outcome):.2f}")
            lines.append(f"  Max odds: {max(all_odds_for_outcome):.2f}")
            lines.append(f"  Mean odds: {sum(all_odds_for_outcome)/len(all_odds_for_outcome):.2f}")
            
            if len(all_odds_for_outcome) >= 2:
                sorted_odds = sorted(all_odds_for_outcome, reverse=True)
                gap = (sorted_odds[0] - sorted_odds[1]) / sorted_odds[1]
                lines.append(f"  Gap to 2nd best: {gap:.1%}")
                
                if gap > 0.20:
                    lines.append(f"  ⚠️  WARNING: Best odds is outlier (>20% gap)!")
                else:
                    lines.append(f"  ✓ Best odds confirmed by market")
                    
                # High odds check
                if bet['odds'] >= 10.0:
                    confirmed = check_high_odds_confirmation(all_odds_for_outcome, 10.0)
                    if confirmed:
                        lines.append(f"  ✓ High odds confirmed by multiple bookmakers")
                    else:
                        lines.append(f"  ⚠️  WARNING: High odds not confirmed!")
        lines.append("")
        
        # Diagnosis
        lines.append("Potential Issues:")
        issues_found = False
        
        # Issue 1: High EV with high odds + high probability
        if bet['odds'] > 10.0 and bet['p_model'] > 0.20:
            lines.append(f"  • High odds ({bet['odds']}) with high model prob ({bet['p_model']:.1%})")
            lines.append(f"    → Model may be overconfident")
            issues_found = True
        
        # Issue 2: Large gap between model and market
        if p_novig and abs(bet['p_model'] - p_novig) > 0.15:
            lines.append(f"  • Large gap: model {bet['p_model']:.1%} vs market {p_novig:.1%}")
            lines.append(f"    → Check if model/market has information advantage")
            issues_found = True
        
        # Issue 3: Probability sum issues
        if event_probs and abs(sum(event_probs.values()) - 1.0) > 0.02:
            lines.append(f"  • Probability sum = {sum(event_probs.values()):.4f}")
            lines.append(f"    → Model probabilities not properly normalized")
            issues_found = True
        
        if not issues_found:
            lines.append(f"  None detected")
        
        lines.append("")
    
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    # Write to file
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return str(report_path)
