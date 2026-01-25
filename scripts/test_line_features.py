"""Test line movement features."""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.odds_tracker import OddsTracker
from src.features.line_movement_features import LineMovementFeatures

def main():
    print("=" * 60)
    print("LINE MOVEMENT FEATURES TEST")
    print("=" * 60)
    
    # Create tracker with test data
    tracker = OddsTracker(db_path='data/odds/test_odds.db')
    
    match_id = "test_match_003"
    commence_time = int(time.time()) + 7200  # 2 hours from now
    
    # Store opening odds
    opening_odds = {
        'Pinnacle': {'home': 2.20, 'draw': 3.30, 'away': 3.40}
    }
    tracker.store_odds_snapshot(match_id, opening_odds, is_opening=True)
    
    # Simulate sharp move
    time.sleep(1)
    
    sharp_odds = {
        'Pinnacle': {'home': 1.90, 'draw': 3.50, 'away': 3.70}  # -13.6% home
    }
    tracker.store_odds_snapshot(match_id, sharp_odds)
    
    # Extract features
    extractor = LineMovementFeatures(tracker)
    features = extractor.extract_for_match(match_id, commence_time)
    
    print("\nExtracted Features:")
    for key, value in features.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Validate
    assert features['home_odds_change_pct'] < -10, "Should detect price drop"
    print("\n✅ Sharp move detected correctly!")

if __name__ == '__main__':
    main()
