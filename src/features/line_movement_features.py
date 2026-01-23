"""
Line movement feature extractor.

Extracts 12 features from odds time-series:
- Opening/current/closing odds (6)
- Movement metrics (6)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging

from ..data.odds_tracker import OddsTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LineMovementFeatures:
    """Extract line movement features for ML model."""
    
    def __init__(self, tracker: Optional[OddsTracker] = None):
        """
        Initialize feature extractor.
        
        Args:
            tracker: OddsTracker instance (creates new if None)
        """
        self.tracker = tracker or OddsTracker()
    
    def extract_for_match(
        self,
        match_id: str,
        commence_time: int
    ) -> Dict:
        """
        Extract all line movement features for a match.
        
        Args:
            match_id: Match identifier
            commence_time: Match start time (Unix timestamp)
            
        Returns:
            Dict with 12 features
        """
        import time
        
        # Get odds snapshots
        opening = self.tracker.get_opening_odds(match_id)
        current = self.tracker.get_current_odds(match_id)
        
        # If no data, return neutral defaults
        if not opening:
            return self._default_features()
        
        if not current:
            current = opening  # Use opening if no updates yet
        
        # Calculate features
        features = {}
        
        # 1-6: Opening and current odds
        features['home_odds_open'] = opening.get('home', 2.0)
        features['draw_odds_open'] = opening.get('draw', 3.0)
        features['away_odds_open'] = opening.get('away', 2.0)
        
        features['home_odds_current'] = current.get('home', 2.0)
        features['draw_odds_current'] = current.get('draw', 3.0)
        features['away_odds_current'] = current.get('away', 2.0)
        
        # 7-8: Percentage change from opening
        features['home_odds_change_pct'] = self._calc_change_pct(
            opening.get('home'), current.get('home')
        )
        features['away_odds_change_pct'] = self._calc_change_pct(
            opening.get('away'), current.get('away')
        )
        
        # 9: Sharp move detection (>10% in <12 hours)
        features['sharp_move_detected'] = self._detect_sharp_move(
            match_id, commence_time
        )
        
        # 10: Odds volatility (average across outcomes)
        features['odds_volatility'] = self._calc_volatility(match_id)
        
        # 11: Time to match (hours)
        hours_to_match = (commence_time - time.time()) / 3600
        features['time_to_match_hours'] = max(hours_to_match, 0)
        
        # 12: Market efficiency (inverse of vig)
        features['market_efficiency_score'] = self._calc_market_efficiency(current)
        
        return features
    
    def _default_features(self) -> Dict:
        """Default features when no odds data available."""
        return {
            'home_odds_open': 2.0,
            'draw_odds_open': 3.0,
            'away_odds_open': 2.0,
            'home_odds_current': 2.0,
            'draw_odds_current': 3.0,
            'away_odds_current': 2.0,
            'home_odds_change_pct': 0.0,
            'away_odds_change_pct': 0.0,
            'sharp_move_detected': 0,
            'odds_volatility': 0.0,
            'time_to_match_hours': 24.0,
            'market_efficiency_score': 0.95
        }
    
    def _calc_change_pct(
        self,
        open_odds: Optional[float],
        current_odds: Optional[float]
    ) -> float:
        """Calculate percentage change in odds."""
        if not open_odds or not current_odds:
            return 0.0
        
        return ((current_odds - open_odds) / open_odds) * 100
    
    def _detect_sharp_move(self, match_id: str, commence_time: int) -> int:
        """
        Detect sharp line movement.
        
        Sharp move criteria:
        - >10% odds change in <12 hours
        
        Returns:
            1 if sharp move detected, 0 otherwise
        """
        import time
        
        # Check home odds movement
        movement = self.tracker.get_line_movement(match_id, 'home')
        
        if len(movement) < 2:
            return 0
        
        # Look for large changes in short periods
        current_time = time.time()
        twelve_hours_ago = current_time - (12 * 3600)
        
        # Filter to last 12 hours
        recent_movement = [(ts, odds) for ts, odds in movement if ts >= twelve_hours_ago]
        
        if len(recent_movement) < 2:
            return 0
        
        # Check max change
        odds_values = [odds for _, odds in recent_movement]
        max_odds = max(odds_values)
        min_odds = min(odds_values)
        
        change_pct = abs((max_odds - min_odds) / max_odds) * 100
        
        return 1 if change_pct > 10 else 0
    
    def _calc_volatility(self, match_id: str) -> float:
        """
        Calculate odds volatility.
        
        Returns average standard deviation across all outcomes.
        """
        volatilities = []
        
        for outcome in ['home', 'draw', 'away']:
            movement = self.tracker.get_line_movement(match_id, outcome)
            
            if len(movement) > 1:
                odds_values = [odds for _, odds in movement]
                vol = float(np.std(odds_values))
                volatilities.append(vol)
        
        if not volatilities:
            return 0.0
        
        return np.mean(volatilities)
    
    def _calc_market_efficiency(self, odds: Dict) -> float:
        """
        Calculate market efficiency score.
        
        Efficiency = 1 / (1/home + 1/draw + 1/away)
        
        Perfect market = 1.0 (no vig)
        Typical market = 0.95-0.97
        """
        if not odds or len(odds) < 3:
            return 0.95
        
        home = odds.get('home', 2.0)
        draw = odds.get('draw', 3.0)
        away = odds.get('away', 2.0)
        
        implied_total = (1/home) + (1/draw) + (1/away)
        
        if implied_total == 0:
            return 0.95
        
        return 1.0 / implied_total


def test_line_features():
    """Test line movement feature extractor."""
    import time
    
    print("=" * 60)
    print("LINE MOVEMENT FEATURES TEST")
    print("=" * 60)
    
    # Create tracker with test data
    tracker = OddsTracker(db_path='data/odds/test_odds.db')
    
    match_id = "test_match_002"
    commence_time = int(time.time()) + 7200  # 2 hours from now
    
    # Store opening odds
    opening_odds = {
        'Pinnacle': {'home': 2.20, 'draw': 3.30, 'away': 3.40}
    }
    tracker.store_odds_snapshot(match_id, opening_odds, is_opening=True)
    
    # Simulate sharp move (home drops to 1.90)
    time.sleep(1)
    
    sharp_odds = {
        'Pinnacle': {'home': 1.90, 'draw': 3.50, 'away': 3.70}  # -13.6% home
    }
    tracker.store_odds_snapshot(match_id, sharp_odds)
    
    # Extract features
    extractor = LineMovementFeatures(tracker)
    features = extractor.extract_for_match(match_id, commence_time)
    
    print("\nExtracted Features:")
    print(f"  home_odds_open: {features['home_odds_open']:.2f}")
    print(f"  home_odds_current: {features['home_odds_current']:.2f}")
    print(f"  home_odds_change_pct: {features['home_odds_change_pct']:+.1f}%")
    print(f"  sharp_move_detected: {features['sharp_move_detected']}")
    print(f"  odds_volatility: {features['odds_volatility']:.4f}")
    print(f"  time_to_match_hours: {features['time_to_match_hours']:.1f}h")
    print(f"  market_efficiency_score: {features['market_efficiency_score']:.4f}")
    
    # Validate
    assert features['home_odds_change_pct'] < -10, "Should detect price drop"
    assert features['sharp_move_detected'] == 1, "Should detect sharp move"
    
    print("\n✅ All validations passed!")


if __name__ == '__main__':
    test_line_features()
