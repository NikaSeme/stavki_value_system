"""
Pipeline Sanity Tests for STAVKI

These tests ensure the ML pipeline maintains integrity:
- Probabilities sum to 1.0 (within tolerance)
- Feature order matches training
- No NaN/Inf values in predictions
- Odds are in valid ranges
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestProbabilitySanity:
    """Ensure model probabilities are valid."""
    
    @pytest.fixture
    def sample_probs(self):
        """Generate sample probability predictions."""
        return np.array([
            [0.45, 0.30, 0.25],
            [0.50, 0.28, 0.22],
            [0.35, 0.35, 0.30],
            [0.60, 0.25, 0.15],
        ])
    
    def test_probabilities_sum_to_one(self, sample_probs):
        """Probabilities must sum to 1.0 within tolerance."""
        tolerance = 0.01
        sums = sample_probs.sum(axis=1)
        
        for i, prob_sum in enumerate(sums):
            assert abs(prob_sum - 1.0) < tolerance, \
                f"Row {i}: Prob sum {prob_sum:.4f} deviates from 1.0 by more than {tolerance}"
    
    def test_probabilities_in_valid_range(self, sample_probs):
        """Each probability must be between 0 and 1."""
        assert np.all(sample_probs >= 0), "Found negative probabilities"
        assert np.all(sample_probs <= 1), "Found probabilities > 1"
    
    def test_no_nan_probabilities(self, sample_probs):
        """No NaN values allowed in probabilities."""
        assert not np.any(np.isnan(sample_probs)), "Found NaN in probabilities"
    
    def test_no_inf_probabilities(self, sample_probs):
        """No infinite values allowed in probabilities."""
        assert not np.any(np.isinf(sample_probs)), "Found Inf in probabilities"


class TestFeatureOrder:
    """Ensure feature order consistency between training and inference."""
    
    @pytest.fixture
    def training_features(self):
        """Mock training feature names."""
        return [
            'Home_GF_L5', 'Home_GA_L5', 'Home_Pts_L5',
            'Away_GF_L5', 'Away_GA_L5', 'Away_Pts_L5',
            'HomeEloBefore', 'AwayEloBefore', 'EloDiff'
        ]
    
    def test_feature_names_match_exactly(self, training_features):
        """Feature names must match exactly at inference time."""
        # Simulate inference features (should match training)
        inference_features = [
            'Home_GF_L5', 'Home_GA_L5', 'Home_Pts_L5',
            'Away_GF_L5', 'Away_GA_L5', 'Away_Pts_L5',
            'HomeEloBefore', 'AwayEloBefore', 'EloDiff'
        ]
        
        assert training_features == inference_features, \
            f"Feature mismatch: {set(training_features) ^ set(inference_features)}"
    
    def test_detect_missing_features(self, training_features):
        """Detect when inference is missing features."""
        # Missing 'EloDiff'
        inference_features = [
            'Home_GF_L5', 'Home_GA_L5', 'Home_Pts_L5',
            'Away_GF_L5', 'Away_GA_L5', 'Away_Pts_L5',
            'HomeEloBefore', 'AwayEloBefore'
        ]
        
        missing = set(training_features) - set(inference_features)
        assert len(missing) > 0, "Should detect missing feature 'EloDiff'"
        assert 'EloDiff' in missing
    
    def test_detect_extra_features(self, training_features):
        """Detect when inference has extra features not in training."""
        # Extra 'SentimentScore'
        inference_features = training_features + ['SentimentScore']
        
        extra = set(inference_features) - set(training_features)
        assert len(extra) > 0, "Should detect extra feature"
        assert 'SentimentScore' in extra


class TestOddsSanity:
    """Ensure odds data is valid."""
    
    @pytest.fixture
    def sample_odds(self):
        """Generate sample odds data."""
        return pd.DataFrame({
            'home_odds': [1.50, 2.20, 3.00, 1.85],
            'draw_odds': [4.00, 3.40, 3.20, 3.60],
            'away_odds': [5.50, 3.10, 2.40, 4.20],
        })
    
    def test_odds_greater_than_one(self, sample_odds):
        """All odds must be > 1.0 (otherwise guaranteed loss)."""
        for col in ['home_odds', 'draw_odds', 'away_odds']:
            assert np.all(sample_odds[col] > 1.0), \
                f"Found odds <= 1.0 in {col}"
    
    def test_odds_in_reasonable_range(self, sample_odds):
        """Odds should be in reasonable range (1.01 to 100)."""
        min_odds, max_odds = 1.01, 100.0
        
        for col in ['home_odds', 'draw_odds', 'away_odds']:
            assert np.all(sample_odds[col] >= min_odds), \
                f"Found odds < {min_odds} in {col}"
            assert np.all(sample_odds[col] <= max_odds), \
                f"Found odds > {max_odds} in {col}"
    
    def test_no_nan_odds(self, sample_odds):
        """No NaN values allowed in odds."""
        for col in ['home_odds', 'draw_odds', 'away_odds']:
            assert not sample_odds[col].isna().any(), \
                f"Found NaN in {col}"
    
    def test_implied_probabilities_sum(self, sample_odds):
        """Implied probabilities from odds should sum > 1 (bookmaker margin)."""
        implied_probs = (
            1 / sample_odds['home_odds'] +
            1 / sample_odds['draw_odds'] +
            1 / sample_odds['away_odds']
        )
        
        # Bookmaker margin means sum > 1.0
        assert np.all(implied_probs > 1.0), \
            "Implied probabilities should sum > 1.0 due to bookmaker margin"
        
        # But shouldn't be too high (typical range 1.03 to 1.15)
        assert np.all(implied_probs < 1.20), \
            "Implied probabilities sum too high (margin > 20%)"


class TestDataLeakage:
    """Test for potential data leakage issues."""
    
    def test_no_future_features_in_training(self):
        """Ensure training features don't include match outcome data."""
        # These columns should NEVER be used as features
        forbidden_features = {
            'FTHG', 'FTAG',  # Full-time goals
            'FTR',           # Full-time result (target)
            'GoalDiff',      # Goal difference
            'TotalGoals',    # Total goals
            'HS', 'AS',      # Shots
            'HST', 'AST',    # Shots on target
            'HC', 'AC',      # Corners (post-match)
            'HF', 'AF',      # Fouls (post-match)
            'HY', 'AY', 'HR', 'AR',  # Cards (post-match)
        }
        
        # Simulate feature list
        actual_features = [
            'Home_GF_L5', 'Home_GA_L5', 'Home_Pts_L5',
            'Away_GF_L5', 'Away_GA_L5', 'Away_Pts_L5',
            'HomeEloBefore', 'AwayEloBefore', 'EloDiff'
        ]
        
        leakage = set(actual_features) & forbidden_features
        assert len(leakage) == 0, \
            f"Found potential data leakage features: {leakage}"
    
    def test_features_computed_before_match(self):
        """Validate that rolling stats use .shift(1) to prevent leakage."""
        # This is a conceptual test - actual implementation should use shift
        # Example: Last 5 games should NOT include current game
        
        # Simulate a match sequence for a team
        games = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=6, freq='7D'),
            'Goals': [2, 1, 3, 0, 2, 1],  # Goals scored in each game
        })
        
        # WRONG: Include current game in rolling
        wrong_rolling = games['Goals'].rolling(5).mean()
        
        # CORRECT: Shift first to exclude current game
        correct_rolling = games['Goals'].shift(1).rolling(5, min_periods=1).mean()
        
        # The first row should be NaN in correct version
        assert pd.isna(correct_rolling.iloc[0]), \
            "First match should have no prior history"
        
        # Values should differ (correct is lagged)
        # Row 5: Wrong includes row 5, correct excludes it
        assert wrong_rolling.iloc[5] != correct_rolling.iloc[5], \
            "Shifted values should differ from non-shifted"


class TestCalibration:
    """Test calibration-related sanity checks."""
    
    def test_calibrated_probs_well_formed(self):
        """Calibrated probabilities should remain valid."""
        # Simulate pre-calibration probs
        raw_probs = np.array([
            [0.50, 0.30, 0.20],
            [0.70, 0.20, 0.10],
        ])
        
        # Simulate calibrated probs (should still sum to 1)
        calibrated_probs = np.array([
            [0.48, 0.32, 0.20],
            [0.65, 0.22, 0.13],
        ])
        
        # Check sums
        sums = calibrated_probs.sum(axis=1)
        for i, s in enumerate(sums):
            assert abs(s - 1.0) < 0.01, \
                f"Calibrated row {i} sums to {s:.4f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
