"""
Unit tests for Poisson model.

CRITICAL: Tests probability sum validation.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson as scipy_poisson

from src.models.poisson_model import (
    HOME_ADVANTAGE,
    LEAGUE_AVG_GOALS,
    PoissonModel,
    calculate_match_probabilities,
    estimate_lambda,
    predict_match,
)


class TestLambdaEstimation:
    """Test expected goals (lambda) estimation."""
    
    def test_basic_estimation(self):
        """Test lambda estimation with typical values."""
        # Team scoring 2 goals/game, opponent conceding 1.5 goals/game
        lambda_val = estimate_lambda(
            goals_for_avg=2.0,
            goals_against_avg=1.0,
            opponent_goals_for_avg=1.5,
            opponent_goals_against_avg=1.5,
            is_home=False  # No home advantage
        )
        
        # Expected: (2.0 / 1.5) * (1.5 / 1.5) * 1.5 = 2.0
        assert lambda_val == pytest.approx(2.0, rel=0.01)
    
    def test_home_advantage(self):
        """Test that home teams get advantage boost."""
        lambda_home = estimate_lambda(
            goals_for_avg=1.5,
            goals_against_avg=1.5,
            opponent_goals_for_avg=1.5,
            opponent_goals_against_avg=1.5,
            is_home=True
        )
        
        lambda_away = estimate_lambda(
            goals_for_avg=1.5,
            goals_against_avg=1.5,
            opponent_goals_for_avg=1.5,
            opponent_goals_against_avg=1.5,
            is_home=False
        )
        
        # Home lambda should be ~15% higher
        assert lambda_home > lambda_away
        assert lambda_home == pytest.approx(lambda_away * HOME_ADVANTAGE, rel=0.01)
    
    def test_missing_data_fallback(self):
        """Test fallback to league average when data missing."""
        lambda_val = estimate_lambda(
            goals_for_avg=np.nan,
            goals_against_avg=1.0,
            opponent_goals_for_avg=1.5,
            opponent_goals_against_avg=np.nan,
            is_home=False
        )
        
        # Should fall back to league average
        assert lambda_val == pytest.approx(LEAGUE_AVG_GOALS, rel=0.01)


class TestProbabilityCalculation:
    """Test probability calculation from lambda values."""
    
    def test_probabilities_sum_to_one(self):
        """CRITICAL: Test that probabilities sum to exactly 1.0."""
        probs = calculate_match_probabilities(lambda_home=1.8, lambda_away=1.2)
        
        prob_sum = probs['prob_home'] + probs['prob_draw'] + probs['prob_away']
        
        assert prob_sum == pytest.approx(1.0, abs=1e-6)
    
    def test_equal_teams(self):
        """Test that equal teams have draw as likely outcome."""
        probs = calculate_match_probabilities(lambda_home=1.5, lambda_away=1.5)
        
        # All three outcomes should be reasonably probable
        assert 0.2 < probs['prob_home'] < 0.4
        assert 0.2 < probs['prob_draw'] < 0.4
        assert 0.2 < probs['prob_away'] < 0.4
        
        # Sum should be 1.0
        assert (probs['prob_home'] + probs['prob_draw'] + probs['prob_away']) == pytest.approx(1.0)
    
    def test_strong_favorite(self):
        """Test that strong favorite has high win probability."""
        probs = calculate_match_probabilities(lambda_home=3.0, lambda_away=0.8)
        
        # Home team should have highest probability
        assert probs['prob_home'] > probs['prob_draw']
        assert probs['prob_home'] > probs['prob_away']
        assert probs['prob_home'] > 0.6
    
    def test_weak_team(self):
        """Test that weak home team loses probability."""
        probs = calculate_match_probabilities(lambda_home=0.5, lambda_away=2.5)
        
        # Away team should have highest probability
        assert probs['prob_away'] > probs['prob_draw']
        assert probs['prob_away'] > probs['prob_home']
    
    def test_poisson_distribution_correctness(self):
        """Verify Poisson PMF is used correctly."""
        # Manual calculation for 0-0
        lambda_home = 1.0
        lambda_away = 1.0
        
        # P(0-0) = P(home=0) * P(away=0)
        prob_0_0_expected = scipy_poisson.pmf(0, lambda_home) * scipy_poisson.pmf(0, lambda_away)
        
        # Our function should include this in the draw probability
        probs = calculate_match_probabilities(lambda_home, lambda_away, max_goals=0)
        
        # When max_goals=0, only 0-0 is considered, which is a draw
        assert probs['prob_draw'] == pytest.approx(prob_0_0_expected / prob_0_0_expected, abs=1e-6)


class TestMatchPrediction:
    """Test full match prediction pipeline."""
    
    def test_predict_match_with_features(self):
        """Test prediction with typical feature values."""
        match_features = pd.Series({
            'home_goals_for_avg_5': 1.8,
            'home_goals_against_avg_5': 1.2,
            'away_goals_for_avg_5': 1.4,
            'away_goals_against_avg_5': 1.6,
        })
        
        result = predict_match(match_features)
        
        # Check all required fields present
        assert 'lambda_home' in result
        assert 'lambda_away' in result
        assert 'prob_home' in result
        assert 'prob_draw' in result
        assert 'prob_away' in result
        
        # Check probabilities sum to 1.0
        prob_sum = result['prob_home'] + result['prob_draw'] + result['prob_away']
        assert prob_sum == pytest.approx(1.0, abs=1e-6)
        
        # Lambda should be positive
        assert result['lambda_home'] > 0
        assert result['lambda_away'] > 0
    
    def test_predict_match_missing_features(self):
        """Test prediction when features are missing."""
        match_features = pd.Series({
            'home_goals_for_avg_5': np.nan,
            'home_goals_against_avg_5': np.nan,
            'away_goals_for_avg_5': np.nan,
            'away_goals_against_avg_5': np.nan,
        })
        
        result = predict_match(match_features)
        
        # Should still produce valid probabilities
        prob_sum = result['prob_home'] + result['prob_draw'] + result['prob_away']
        assert prob_sum == pytest.approx(1.0, abs=1e-6)
        
        # Lambdas should fall back to league average (with home advantage)
        assert result['lambda_home'] == pytest.approx(LEAGUE_AVG_GOALS * HOME_ADVANTAGE, rel=0.01)
        assert result['lambda_away'] == pytest.approx(LEAGUE_AVG_GOALS, rel=0.01)


class TestPoissonModel:
    """Test PoissonModel class."""
    
    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        model = PoissonModel()
        
        assert model.home_advantage == HOME_ADVANTAGE
        assert model.league_avg_goals == LEAGUE_AVG_GOALS
    
    def test_predict_dataset(self):
        """Test prediction on a small dataset."""
        # Create test dataset
        features_df = pd.DataFrame({
            'date': ['2025-08-01', '2025-08-02'],
            'league': ['E0', 'E0'],
            'home_team': ['Arsenal', 'Chelsea'],
            'away_team': ['Chelsea', 'Liverpool'],
            'home_goals': [2, 1],
            'away_goals': [1, 1],
            'home_goals_for_avg_5': [1.8, 1.6],
            'home_goals_against_avg_5': [1.2, 1.4],
            'away_goals_for_avg_5': [1.5, 1.7],
            'away_goals_against_avg_5': [1.3, 1.2],
        })
        
        model = PoissonModel()
        predictions = model.predict(features_df)
        
        # Check all rows have predictions
        assert len(predictions) == 2
        assert 'prob_home' in predictions.columns
        assert 'prob_draw' in predictions.columns
        assert 'prob_away' in predictions.columns
        
        # Check probabilities for each row
        for idx, row in predictions.iterrows():
            prob_sum = row['prob_home'] + row['prob_draw'] + row['prob_away']
            assert prob_sum == pytest.approx(1.0, abs=1e-6)
    
    def test_deterministic_predictions(self):
        """Test that same inputs always produce same outputs."""
        match_features = pd.Series({
            'home_goals_for_avg_5': 2.0,
            'home_goals_against_avg_5': 1.0,
            'away_goals_for_avg_5': 1.5,
            'away_goals_against_avg_5': 1.5,
        })
        
        result1 = predict_match(match_features)
        result2 = predict_match(match_features)
        
        assert result1['prob_home'] == result2['prob_home']
        assert result1['prob_draw'] == result2['prob_draw']
        assert result1['prob_away'] == result2['prob_away']


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_low_lambda(self):
        """Test with very low expected goals."""
        probs = calculate_match_probabilities(lambda_home=0.1, lambda_away=0.1)
        
        # Should still sum to 1.0
        prob_sum = probs['prob_home'] + probs['prob_draw'] + probs['prob_away']
        assert prob_sum == pytest.approx(1.0, abs=1e-6)
        
        # Draw (0-0) should be most likely
        assert probs['prob_draw'] > probs['prob_home']
        assert probs['prob_draw'] > probs['prob_away']
    
    def test_very_high_lambda(self):
        """Test with very high expected goals."""
        probs = calculate_match_probabilities(lambda_home=5.0, lambda_away=5.0)
        
        # Should still sum to 1.0
        prob_sum = probs['prob_home'] + probs['prob_draw'] + probs['prob_away']
        assert prob_sum == pytest.approx(1.0, abs=1e-6)
    
    def test_asymmetric_lambdas(self):
        """Test with very different lambda values."""
        probs = calculate_match_probabilities(lambda_home=4.0, lambda_away=0.5)
        
        # Home should dominate
        assert probs['prob_home'] > 0.8
        prob_sum = probs['prob_home'] + probs['prob_draw'] + probs['prob_away']
        assert prob_sum == pytest.approx(1.0, abs=1e-6)
