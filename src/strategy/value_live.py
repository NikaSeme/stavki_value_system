"""
Live value bet finder module.

Loads latest odds, computes EV using model probabilities, and ranks value opportunities.
"""

from __future__ import annotations

import csv
import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .no_vig import implied_prob_from_decimal, no_vig_proportional
from .value import compute_ev


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


def select_best_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each event+market+outcome, select the bookmaker offering the best (highest) price.
    
    Args:
        df: DataFrame with columns: event_id, market_key, outcome_name, outcome_price, bookmaker_title
        
    Returns:
        DataFrame with best prices only
    """
    # Group by event, market, outcome and pick max price
    idx = df.groupby(['event_id', 'market_key', 'outcome_name'])['outcome_price'].idxmax()
    return df.loc[idx].reset_index(drop=True)


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


def get_model_probabilities(
    events: pd.DataFrame,
    model_type: str = "simple"
) -> Dict[str, Dict[str, float]]:
    """
    Get model-based probabilities for each event.
    
    For MVP: Use simple Poisson/ELO-based estimates.
    Future: Integrate full ensemble model with feature engineering.
    
    Args:
        events: DataFrame with event details (home_team, away_team, etc.)
        model_type: 'simple' for baseline model
        
    Returns:
        Dict[event_id][outcome_name] = model probability
    """
    if model_type == "simple":
        return _get_simple_model_probs(events)
    else:
        raise NotImplementedError(f"Model type '{model_type}' not implemented")


def _get_simple_model_probs(events: pd.DataFrame) -> Dict[str, Dict[str, float]]:
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
        
        # Simple baseline: 45% home, 30% away, 25% draw for soccer
        # Adjust based on sport if needed
        probs[event_id] = {
            home_team: 0.45,
            away_team: 0.30,
        }
        
        # Check if this market has draws (3-way)
        if 'Draw' in event.get('outcome_name', ''):
            probs[event_id]['Draw'] = 0.25
            # Normalize
            total = sum(probs[event_id].values())
            probs[event_id] = {k: v/total for k, v in probs[event_id].items()}
    
    return probs


def compute_ev_candidates(
    model_probs: Dict[str, Dict[str, float]],
    best_prices: pd.DataFrame,
    threshold: float = 0.05,
    market_key: str = "h2h"
) -> List[Dict[str, Any]]:
    """
    Compute expected value for all bets and filter by threshold.
    
    Args:
        model_probs: Model probabilities per event and outcome
        best_prices: Best odds per outcome
        threshold: Minimum EV to include
        market_key: Market to analyze (e.g., 'h2h')
        
    Returns:
        List of value bet candidates with EV, odds, probabilities
    """
    candidates = []
    
    # Filter to specified market
    market_df = best_prices[best_prices['market_key'] == market_key]
    
    for _, row in market_df.iterrows():
        event_id = row['event_id']
        outcome = row['outcome_name']
        odds = float(row['outcome_price'])
        
        # Get model probability for this outcome
        event_probs = model_probs.get(event_id, {})
        p_model = event_probs.get(outcome)
        
        if p_model is None:
            continue
        
        # Calculate EV
        ev = compute_ev(p_model, odds)
        
        if ev >= threshold:
            candidates.append({
                'event_id': event_id,
                'sport_key': row.get('sport_key', ''),
                'commence_time': row.get('commence_time', ''),
                'home_team': row.get('home_team', ''),
                'away_team': row.get('away_team', ''),
                'market': market_key,
                'selection': outcome,
                'odds': odds,
                'bookmaker': row['bookmaker_title'],
                'bookmaker_key': row.get('bookmaker_key', ''),
                'p_model': round(p_model, 4),
                'p_implied': round(implied_prob_from_decimal(odds), 4),
                'ev': round(ev, 4),
                'ev_pct': round(ev * 100, 2),
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
