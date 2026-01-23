"""
Integration tests for ML model in live value pipeline.

Tests end-to-end flow with ML model to ensure:
- ML model loads correctly
- No baseline fallback
- Probabilities sum to 1.0
- EVs are within expected ranges
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy.value_live import (
    initialize_ml_model,
    get_model_probabilities,
    is_baseline_model_output,
)
from src.models import ModelLoader
from src.features.live_extractor import LiveFeatureExtractor


class TestMLLiveIntegration:
    """Test ML model integration in live pipeline."""
    
    def test_ml_model_initializes(self):
        """Test that ML model initializes without errors."""
        # Should not raise
        initialize_ml_model()
    
    def test_ml_model_not_baseline(self):
        """Test that ML model does not produce baseline patterns."""
        initialize_ml_model()
        
        # Create test events
        events = pd.DataFrame([
            {
                'event_id': 'test1',
                'sport_key': 'soccer_epl',
                'home_team': 'Arsenal',
                'away_team': 'Chelsea',
                'commence_time': '2026-01-24T15:00:00Z'
            },
            {
                'event_id': 'test2',
                'sport_key': 'soccer_epl',
                'home_team': 'Manchester City',
                'away_team': 'Liverpool',
                'commence_time': '2026-01-24T17:30:00Z'
            }
        ])
        
        # Create test odds
        odds = pd.DataFrame([
            {'event_id': 'test1', 'outcome_name': 'Arsenal', 'outcome_price': 2.0},
            {'event_id': 'test1', 'outcome_name': 'Draw', 'outcome_price': 3.5},
            {'event_id': 'test1', 'outcome_name': 'Chelsea', 'outcome_price': 3.0},
            {'event_id': 'test2', 'outcome_name': 'Manchester City', 'outcome_price': 1.5},
            {'event_id': 'test2', 'outcome_name': 'Draw', 'outcome_price': 4.0},
            {'event_id': 'test2', 'outcome_name': 'Liverpool', 'outcome_price': 5.0},
        ])
        
        # Get model probabilities
        probs = get_model_probabilities(events, odds_df=odds, model_type='ml')
        
        # Check NOT baseline
        is_baseline = is_baseline_model_output(probs)
        assert not is_baseline, "ML model should not produce baseline pattern"
    
    def test_ml_probabilities_sum_to_one(self):
        """Test that ML probabilities sum to 1.0 for each event."""
        initialize_ml_model()
        
        events = pd.DataFrame([{
            'event_id': 'test123',
            'sport_key': 'soccer_epl',
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
            'commence_time': '2026-01-24T15:00:00Z'
        }])
        
        odds = pd.DataFrame([
            {'event_id': 'test123', 'outcome_name': 'Arsenal', 'outcome_price': 2.0},
            {'event_id': 'test123', 'outcome_name': 'Draw', 'outcome_price': 3.5},
            {'event_id': 'test123', 'outcome_name': 'Chelsea', 'outcome_price': 3.0},
        ])
        
        probs = get_model_probabilities(events, odds_df=odds, model_type='ml')
        
        # Check sum for each event
        for event_id, event_probs in probs.items():
            prob_sum = sum(event_probs.values())
            assert abs(prob_sum - 1.0) < 0.01, f"Probabilities should sum to 1.0, got {prob_sum}"
    
    def test_ml_probabilities_in_valid_range(self):
        """Test that ML probabilities are in [0, 1]."""
        initialize_ml_model()
        
        events = pd.DataFrame([{
            'event_id': 'test456',
            'sport_key': 'soccer_epl',
            'home_team': 'Liverpool',
            'away_team': 'Manchester United',
            'commence_time': '2026-01-24T15:00:00Z'
        }])
        
        odds = pd.DataFrame([
            {'event_id': 'test456', 'outcome_name': 'Liverpool', 'outcome_price': 1.8},
            {'event_id': 'test456', 'outcome_name': 'Draw', 'outcome_price': 3.5},
            {'event_id': 'test456', 'outcome_name': 'Manchester United', 'outcome_price': 4.0},
        ])
        
        probs = get_model_probabilities(events, odds_df=odds, model_type='ml')
        
        # Check all probabilities in valid range
        for event_id, event_probs in probs.items():
            for outcome, prob in event_probs.items():
                assert 0.0 <= prob <= 1.0, f"Probability {prob} out of valid range [0,1]"
    
    def test_baseline_detection_works(self):
        """Test that baseline detection correctly identifies baseline patterns."""
        # Get baseline probabilities
        events = pd.DataFrame([
            {'event_id': 'b1', 'sport_key': 'soccer_epl', 'home_team': 'Team A', 'away_team': 'Team B'},
            {'event_id': 'b2', 'sport_key': 'soccer_epl', 'home_team': 'Team C', 'away_team': 'Team D'},
        ])
        
        baseline_probs = get_model_probabilities(events, model_type='simple')
        
        # Should detect baseline
        is_baseline = is_baseline_model_output(baseline_probs)
        assert is_baseline, "Should detect baseline pattern"
    
    def test_ml_model_produces_different_probabilities(self):
        """Test that ML model produces different probabilities for different matchups."""
        initialize_ml_model()
        
        # Create events with very different expected outcomes
        events = pd.DataFrame([
            {
                'event_id': 'strong_favorite',
                'sport_key': 'soccer_epl',
                'home_team': 'Manchester City',
                'away_team': 'Burnley',
                'commence_time': '2026-01-24T15:00:00Z'
            },
            {
                'event_id': 'even_matchup',
                'sport_key': 'soccer_epl',
                'home_team': 'Arsenal',
                'away_team': 'Chelsea',
                'commence_time': '2026-01-24T17:30:00Z'
            }
        ])
        
        odds = pd.DataFrame([
            {'event_id': 'strong_favorite', 'outcome_name': 'Manchester City', 'outcome_price': 1.2},
            {'event_id': 'strong_favorite', 'outcome_name': 'Draw', 'outcome_price': 6.0},
            {'event_id': 'strong_favorite', 'outcome_name': 'Burnley', 'outcome_price': 15.0},
            {'event_id': 'even_matchup', 'outcome_name': 'Arsenal', 'outcome_price': 2.4},
            {'event_id': 'even_matchup', 'outcome_name': 'Draw', 'outcome_price': 3.3},
            {'event_id': 'even_matchup', 'outcome_name': 'Chelsea', 'outcome_price': 3.0},
        ])
        
        probs = get_model_probabilities(events, odds_df=odds, model_type='ml')
        
        # Check that probabilities are different (not fixed like baseline)
        home_probs = [probs[eid][events.loc[events['event_id']==eid, 'home_team'].iloc[0]] 
                      for eid in probs.keys()]
        
        # Should not all be the same (like baseline's 0.40)
        assert len(set([round(p, 2) for p in home_probs])) > 1, \
            "ML model should produce different probabilities for different matchups"
    
    def test_model_loader_validation(self):
        """Test that ModelLoader validates properly."""
        loader = ModelLoader()
        success = loader.load_latest()
        
        assert success, "Model should load successfully"
        assert loader.validate(), "Model should validate"
        assert loader.model is not None
        assert loader.calibrator is not None
        assert loader.scaler is not None
    
    def test_feature_extractor_shape(self):
        """Test that feature extractor produces correct shape."""
        extractor = LiveFeatureExtractor()
        
        events = pd.DataFrame([{
            'event_id': 'test',
            'home_team': 'Arsenal',
            'away_team': 'Chelsea'
        }])
        
        odds = pd.DataFrame([
            {'event_id': 'test', 'outcome_name': 'Arsenal', 'outcome_price': 2.0},
            {'event_id': 'test', 'outcome_name': 'Draw', 'outcome_price': 3.5},
            {'event_id': 'test', 'outcome_name': 'Chelsea', 'outcome_price': 3.0},
        ])
        
        features = extractor.extract_features(events, odds)
        
        # Should be (1, 22) - 1 event, 22 features
        assert features.shape == (1, 22), f"Expected shape (1, 22), got {features.shape}"
    
    def test_ml_requires_odds_df(self):
        """Test that ML mode requires odds_df parameter."""
        initialize_ml_model()
        
        events = pd.DataFrame([{
            'event_id': 'test',
            'sport_key': 'soccer_epl',
            'home_team': 'Arsenal',
            'away_team': 'Chelsea'
        }])
        
        # Should raise ValueError without odds_df
        with pytest.raises(ValueError, match="odds_df required"):
            get_model_probabilities(events, odds_df=None, model_type='ml')


def test_ev_sanity_bounds():
    """Test that EVs from ML model are within reasonable bounds."""
    initialize_ml_model()
    
    events = pd.DataFrame([{
        'event_id': 'sanity_test',
        'sport_key': 'soccer_epl',
        'home_team': 'Arsenal',
        'away_team': 'Chelsea',
        'commence_time': '2026-01-24T15:00:00Z'
    }])
    
    odds = pd.DataFrame([
        {'event_id': 'sanity_test', 'outcome_name': 'Arsenal', 'outcome_price': 2.0},
        {'event_id': 'sanity_test', 'outcome_name': 'Draw', 'outcome_price': 3.5},
        {'event_id': 'sanity_test', 'outcome_name': 'Chelsea', 'outcome_price': 3.0},
    ])
    
    probs = get_model_probabilities(events, odds_df=odds, model_type='ml')
    
    # Calculate EVs
    for event_id, event_probs in probs.items():
        for outcome, p_model in event_probs.items():
            # Get odds for this outcome
            outcome_odds = odds[odds['outcome_name'] == outcome]['outcome_price'].iloc[0]
            
            # Calculate EV
            ev = p_model * outcome_odds - 1
            
            # Sanity bounds: EV should rarely exceed +200% or be below -90%
            # (These are very loose bounds for safety)
            assert ev > -0.95, f"EV too negative: {ev:.2%} for {outcome}"
            assert ev < 3.0, f"EV too high: {ev:.2%} for {outcome} (may indicate model error)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
