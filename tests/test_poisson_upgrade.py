
import pytest
import numpy as np
import pandas as pd
from src.models.poisson_model import (
    estimate_lambda,
    predict_match,
    load_league_config,
    DEFAULT_HOME_ADVANTAGE,
    LEAGUE_AVG_GOALS
)

def test_estimate_lambda_defaults():
    """Test standard lambda calculation without Elo."""
    # Equal teams, home advantage applied
    lam = estimate_lambda(
        goals_for_avg=1.5,
        goals_against_avg=1.5,
        opponent_goals_for_avg=1.5,
        opponent_goals_against_avg=1.5,
        is_home=True,
        home_advantage=1.15
    )
    # Expected: 1.5 * 1.0 * 1.0 * 1.15 = 1.725
    assert abs(lam - 1.725) < 0.001

    # Away team, no advantage
    lam_away = estimate_lambda(
        goals_for_avg=1.5,
        goals_against_avg=1.5,
        opponent_goals_for_avg=1.5,
        opponent_goals_against_avg=1.5,
        is_home=False,
        home_advantage=1.15
    )
    # Expected: 1.5 * 1.0 * 1.0 = 1.5
    assert abs(lam_away - 1.5) < 0.001

def test_estimate_lambda_with_elo():
    """Test lambda adjustment based on Elo ratings."""
    # Strong team (Elo 1600 vs Base 1500) -> +10% boost
    lam_strong = estimate_lambda(
        goals_for_avg=1.5,
        goals_against_avg=1.5,
        opponent_goals_for_avg=1.5,
        opponent_goals_against_avg=1.5,
        is_home=False,  # exclude HA to isolate Elo
        elo_rating=1600
    )
    # Multiplier: 1.0 + (1600-1500)/1000 = 1.10
    # Expected: 1.5 * 1.10 = 1.65
    assert abs(lam_strong - 1.65) < 0.001

    # Weak team (Elo 1400 vs Base 1500) -> -10% penalty
    lam_weak = estimate_lambda(
        goals_for_avg=1.5,
        goals_against_avg=1.5,
        opponent_goals_for_avg=1.5,
        opponent_goals_against_avg=1.5,
        is_home=False,
        elo_rating=1400
    )
    # Multiplier: 1.0 + (1400-1500)/1000 = 0.90
    # Expected: 1.5 * 0.90 = 1.35
    assert abs(lam_weak - 1.35) < 0.001

def test_league_config_loading():
    """Test that league configuration can be loaded."""
    config = load_league_config()
    assert isinstance(config, dict)
    # EPL should have 1.15 based on our task
    if 'soccer_epl' in config:
        assert config['soccer_epl'] == 1.15

def test_predict_match_integration():
    """Test end-to-end prediction with config map."""
    ha_map = {'test_league': 1.25}
    
    features = pd.Series({
        'home_goals_for_avg_5': 1.5,
        'home_goals_against_avg_5': 1.5,
        'away_goals_for_avg_5': 1.5,
        'away_goals_against_avg_5': 1.5,
        'HomeEloBefore': 1500,
        'AwayEloBefore': 1500,
        'League': 'test_league'  # Should trigger 1.25 HA
    })
    
    result = predict_match(features, ha_map=ha_map)
    
    # Verify metadata passed through
    assert result['used_home_advantage'] == 1.25
    
    # Calculate expected home lambda: 1.5 * 1.25 (HA) * 1.0 (Elo) = 1.875
    assert abs(result['lambda_home'] - 1.875) < 0.001

def test_missing_data_fallback():
    """Test fallback when data is missing but Elo is present."""
    lam = estimate_lambda(
        goals_for_avg=np.nan,
        goals_against_avg=np.nan,
        opponent_goals_for_avg=np.nan,
        opponent_goals_against_avg=np.nan,
        is_home=False,
        elo_rating=1600 # Should still apply boost
    )
    # Base 1.5 * 1.10 (Elo) = 1.65
    assert abs(lam - 1.65) < 0.001
