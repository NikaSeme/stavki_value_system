"""
Tests for value bet guardrails and validation.
"""

import pandas as pd
import pytest

from src.strategy.value_live import (
    normalize_team_name,
    validate_outcome_mapping,
    validate_prob_sum,
    renormalize_probs,
    check_outlier_odds,
    check_high_odds_confirmation,
    select_best_prices,
)


def test_normalize_team_name():
    """Test team name normalization."""
    assert normalize_team_name("Manchester City") == "manchester city"
    assert normalize_team_name("Man City") == "manchester city"
    assert normalize_team_name("Wolves") == "wolverhampton wanderers"
    assert normalize_team_name("West Ham United  ") == "west ham united"
    assert normalize_team_name("Brighton & Hove Albion") == "brighton hove albion"


def test_validate_outcome_mapping():
    """Test outcome to probability mapping."""
    model_probs = {
        "Arsenal": 0.45,
        "Chelsea": 0.30,
        "Draw": 0.25,
    }
    
    # Valid mappings
    assert validate_outcome_mapping("Arsenal", "Arsenal", "Chelsea", model_probs) == 0.45
    assert validate_outcome_mapping("Chelsea", "Arsenal", "Chelsea", model_probs) == 0.30
    assert validate_outcome_mapping("Draw", "Arsenal", "Chelsea", model_probs) == 0.25
    
    # Case variations
    assert validate_outcome_mapping("arsenal", "Arsenal", "Chelsea", model_probs) == 0.45
    
    # Invalid mapping
    assert validate_outcome_mapping("Liverpool", "Arsenal", "Chelsea", model_probs) is None


def test_validate_prob_sum():
    """Test probability sum validation."""
    # Valid sums
    assert validate_prob_sum({"a": 0.5, "b": 0.5}, tolerance=0.02) is True
    assert validate_prob_sum({"a": 0.33, "b": 0.33, "c": 0.34}, tolerance=0.02) is True
    
    # Just at edge of tolerance
    assert validate_prob_sum({"a": 0.49, "b": 0.49}, tolerance=0.02) is False  # Sum = 0.98, diff = 0.02, not <=
    assert validate_prob_sum({"a": 0.505, "b": 0.505}, tolerance=0.02) is True  # Sum = 1.01, diff = 0.01, within
    
    # Outside tolerance
    assert validate_prob_sum({"a": 0.4, "b": 0.4}, tolerance=0.02) is False  # Sum = 0.8, diff = 0.2
    assert validate_prob_sum({"a": 0.6, "b": 0.6}, tolerance=0.02) is False  # Sum = 1.2, diff = 0.2


def test_renormalize_probs():
    """Test probability renormalization."""
    # Normal case
    probs = {"a": 0.6, "b": 0.6}
    result = renormalize_probs(probs)
    assert abs(result["a"] - 0.5) < 0.001
    assert abs(result["b"] - 0.5) < 0.001
    
    # Already normalized
    probs = {"a": 0.5, "b": 0.5}
    result = renormalize_probs(probs)
    assert abs(result["a"] - 0.5) < 0.001


def test_check_outlier_odds():
    """Test outlier odds detection."""
    # No outlier
    assert check_outlier_odds([2.00, 1.95, 1.90], gap_threshold=0.20) is False
    
    # Outlier (>20% gap)
    assert check_outlier_odds([3.0, 2.0, 1.95], gap_threshold=0.20) is True
    
    # Edge case: exactly 20% gap
    assert check_outlier_odds([2.4, 2.0], gap_threshold=0.20) is False
    assert check_outlier_odds([2.41, 2.0], gap_threshold=0.20) is True
    
    # Single bookmaker
    assert check_outlier_odds([2.0], gap_threshold=0.20) is False


def test_check_high_odds_confirmation():
    """Test high odds confirmation logic."""
    # Not high odds
    assert check_high_odds_confirmation([5.0, 4.9, 4.8], odds_threshold=10.0) is True
    
    # High odds with confirmation (multiple similar)
    assert check_high_odds_confirmation([12.0, 11.5, 11.8], odds_threshold=10.0) is True
    
    # High odds without confirmation (single outlier)
    assert check_high_odds_confirmation([15.0, 10.0, 9.5], odds_threshold=10.0) is False
    
    # High odds with 2 similar bookmakers
    assert check_high_odds_confirmation([12.0, 11.0], odds_threshold=10.0, similarity_threshold=0.10) is True


def test_select_best_prices_simple():
    """Test simple best price selection."""
    df = pd.DataFrame([
        {
            'event_id': 'evt1',
            'market_key': 'h2h',
            'outcome_name': 'Arsenal',
            'outcome_price': 2.00,
            'bookmaker_title': 'Bet365',
        },
        {
            'event_id': 'evt1',
            'market_key': 'h2h',
            'outcome_name': 'Arsenal',
            'outcome_price': 2.10,
            'bookmaker_title': 'Pinnacle',
        },
    ])
    
    result = select_best_prices(df, check_outliers=False)
    
    assert len(result) == 1
    assert result.iloc[0]['outcome_price'] == 2.10
    assert result.iloc[0]['bookmaker_title'] == 'Pinnacle'


def test_select_best_prices_with_outlier_detection():
    """Test best price selection with outlier filtering."""
    df = pd.DataFrame([
        {
            'event_id': 'evt1',
            'market_key': 'h2h',
            'outcome_name': 'Arsenal',
            'outcome_price': 2.00,
            'bookmaker_title': 'Bet365',
        },
        {
            'event_id': 'evt1',
            'market_key': 'h2h',
            'outcome_name': 'Arsenal',
            'outcome_price': 2.05,
            'bookmaker_title': 'Pinnacle',
        },
        {
            'event_id': 'evt1',
            'market_key': 'h2h',
            'outcome_name': 'Arsenal',
            'outcome_price': 3.00,  # Outlier: 46% above second-best
            'bookmaker_title': 'Outlier',
        },
    ])
    
    result = select_best_prices(df, check_outliers=True, outlier_gap=0.20)
    
    assert len(result) == 1
    assert result.iloc[0]['outcome_price'] == 2.05  # Second-best, not outlier
    assert result.iloc[0]['bookmaker_title'] == 'Pinnacle'


def test_select_best_prices_no_outlier():
    """Test that non-outlier prices are not filtered."""
    df = pd.DataFrame([
        {
            'event_id': 'evt1',
            'market_key': 'h2h',
            'outcome_name': 'Arsenal',
            'outcome_price': 2.00,
            'bookmaker_title': 'Bet365',
        },
        {
            'event_id': 'evt1',
            'market_key': 'h2h',
            'outcome_name': 'Arsenal',
            'outcome_price': 2.10,  # 5% above best, not outlier
            'bookmaker_title': 'Pinnacle',
        },
    ])
    
    result = select_best_prices(df, check_outliers=True, outlier_gap=0.20)
    
    assert len(result) == 1
    assert result.iloc[0]['outcome_price'] == 2.10  # Best price kept
