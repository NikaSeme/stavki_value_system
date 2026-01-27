
import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy import value_live
from src.strategy.value_live import get_model_probabilities

# Mock classes
class MockScaler:
    def __init__(self, n_features):
        self.n_features_in_ = n_features

class MockLoader:
    def __init__(self, features):
        self.features = features
        self.scaler = MockScaler(len(features))
    
    def get_feature_names(self):
        return self.features
    
    def predict(self, X):
        # Return dummy probabilities
        return np.array([[0.33, 0.33, 0.34]] * len(X))

class MockExtractor:
    def extract_features(self, events, odds):
        # Return a DataFrame with a fixed set of "live" features (e.g., 22 standard ones)
        # This simulates the extractor not knowing about new model requirements
        cols = [f"feat_{i}" for i in range(22)]
        return pd.DataFrame(np.zeros((len(events), 22)), columns=cols)

class TestFeatureAlignment(unittest.TestCase):
    def setUp(self):
        # Inject mocks into value_live
        value_live._ml_feature_extractor = MockExtractor()
        
    def test_alignment_extra_columns(self):
        """Test model expecting FEWER features than extracted (dropping extras)."""
        # Model expects first 10 features only
        expected_feats = [f"feat_{i}" for i in range(10)]
        value_live._ml_model_loader = MockLoader(expected_feats)
        
        events = pd.DataFrame([{'event_id': '1', 'home_team': 'H', 'away_team': 'A'}])
        odds = pd.DataFrame([{'event_id': '1'}])
        
        # Should not raise error
        probs = get_model_probabilities(events, odds, model_type="ml")
        self.assertTrue(len(probs) == 1)
        print("\n✅ test_alignment_extra_columns passed")

    def test_alignment_missing_columns(self):
        """Test model expecting MORE features than extracted (adding missing)."""
        # Model expects 25 features (Standard 22 + 3 new ones)
        expected_feats = [f"feat_{i}" for i in range(25)]
        value_live._ml_model_loader = MockLoader(expected_feats)
        
        events = pd.DataFrame([{'event_id': '1', 'home_team': 'H', 'away_team': 'A'}])
        odds = pd.DataFrame([{'event_id': '1'}])
        
        # Should not raise error
        probs = get_model_probabilities(events, odds, model_type="ml")
        self.assertTrue(len(probs) == 1)
        print("\n✅ test_alignment_missing_columns passed")

    def test_alignment_reordering(self):
        """Test columns being reordered to match model expectation."""
        # Model expects reversed order
        expected_feats = [f"feat_{i}" for i in range(21, -1, -1)]
        value_live._ml_model_loader = MockLoader(expected_feats)
        
        events = pd.DataFrame([{'event_id': '1', 'home_team': 'H', 'away_team': 'A'}])
        odds = pd.DataFrame([{'event_id': '1'}])
        
        probs = get_model_probabilities(events, odds, model_type="ml")
        self.assertTrue(len(probs) == 1)
        print("\n✅ test_alignment_reordering passed")

if __name__ == '__main__':
    unittest.main()
