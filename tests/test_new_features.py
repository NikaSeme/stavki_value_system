
import pytest
import pandas as pd
from src.features.live_extractor import LiveFeatureExtractor

class MockLineMovement:
    def extract_for_match(self, match_id, commence_time):
        return {
            'sharp_move_detected': 1,
            'odds_volatility': 0.05,
            'time_to_match_hours': 12.0,
            'market_efficiency_score': 0.98
        }

def test_momentum_features():
    """Test win/loss streak calculation."""
    extractor = LiveFeatureExtractor()
    team = 'Arsenal'
    
    # Simulate history: Win, Win, Loss, Win, Win
    # Current streaks based on history (reversed order in _calculate_streak logic usually implies processing)
    # But let's check internal update logic
    
    # Add matches manually to state
    extractor.team_form[team] = [
        {'won': 1, 'points': 3, 'clean_sheet': 1}, # Oldest
        {'won': 0, 'points': 0, 'clean_sheet': 0}, # Loss
        {'won': 1, 'points': 3, 'clean_sheet': 0}, # Win
        {'won': 1, 'points': 3, 'clean_sheet': 1}, # Win (Newest)
    ]
    
    # Check streaks
    win_streak = extractor._calculate_streak(team, 'won')
    loss_streak = extractor._calculate_streak(team, 'loss')
    
    # Should be 2 wins in a row from newest side
    assert win_streak == 2
    assert loss_streak == 0
    
    # Add a loss
    extractor.team_form[team].append({'won': 0, 'points': 0, 'clean_sheet': 0})
    
    # Check again
    win_streak = extractor._calculate_streak(team, 'won')
    loss_streak = extractor._calculate_streak(team, 'loss')
    
    assert win_streak == 0
    assert loss_streak == 1

def test_line_feature_integration(monkeypatch):
    """Test that line features are integrated into main extract_features."""
    
    # Mock Line Extractor
    extractor = LiveFeatureExtractor()
    extractor.line_extractor = MockLineMovement()
    
    # Create test event
    events = pd.DataFrame([{
        'event_id': 'test_1',
        'home_team': 'Arsenal',
        'away_team': 'Chelsea',
        'league': 'soccer_epl'
    }])
    
    odds = pd.DataFrame([
        {'event_id': 'test_1', 'outcome_name': 'Arsenal', 'outcome_price': 2.0},
        {'event_id': 'test_1', 'outcome_name': 'Draw', 'outcome_price': 3.5},
        {'event_id': 'test_1', 'outcome_name': 'Chelsea', 'outcome_price': 3.0},
    ])
    
    features = extractor.extract_features(events, odds)
    
    assert 'sharp_move_detected' in features.columns
    assert features.iloc[0]['sharp_move_detected'] == 1
    assert features.iloc[0]['market_efficiency_score'] == 0.98

def test_sentiment_graceful_degradation():
    """Test that missing sentiment extractor doesn't crash."""
    extractor = LiveFeatureExtractor()
    extractor.sentiment_extractor = None  # Simulate failure
    
    events = pd.DataFrame([{
        'event_id': 'test_1',
        'home_team': 'Arsenal',
        'away_team': 'Chelsea'
    }])
    
    # Should log warning but verify empty dict returned
    sent = extractor.fetch_sentiment_for_events(events)
    assert sent == {}
    
def test_fatigue_feature():
    """Test days since last match feature."""
    extractor = LiveFeatureExtractor()
    # Default is 7.0
    val = extractor._days_since_last_match('NewTeam')
    assert val == 7.0
