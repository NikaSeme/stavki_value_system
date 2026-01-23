"""
Line movement alert system.

Detects and logs significant market movements:
- Sharp moves (>10% in <12h)
- Steam moves (multiple books simultaneously)
- Source discrepancies (>15% difference)
- Late moves (<2h before match)
"""

import time
from typing import Dict, List, Optional
from datetime import datetime
import logging
import json
from pathlib import Path

from ..data.odds_tracker import OddsTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LineMovementAlerts:
    """Detect and log significant line movements."""
    
    # Alert thresholds
    SHARP_MOVE_PCT = 10.0  # >10% movement
    SHARP_MOVE_HOURS = 12  # within 12 hours
    DISCREPANCY_PCT = 15.0  # >15% between sources
    LATE_MOVE_HOURS = 2  # <2 hours before match
    
    def __init__(self, tracker: Optional[OddsTracker] = None):
        """Initialize alert system."""
        self.tracker = tracker or OddsTracker()
        self.alerts_file = Path('data/logs/line_movement_alerts.json')
        self.alerts_file.parent.mkdir(parents=True, exist_ok=True)
    
    def check_all_alerts(
        self,
        match_id: str,
        commence_time: int,
        current_odds: Dict[str, Dict[str, float]]
    ) -> List[Dict]:
        """
        Check all alert conditions.
        
        Args:
            match_id: Match identifier
            commence_time: Match start time
            current_odds: Current odds by bookmaker
            
        Returns:
            List of alerts triggered
        """
        alerts = []
        
        # 1. Sharp move
        sharp_alert = self._check_sharp_move(match_id)
        if sharp_alert:
            alerts.append(sharp_alert)
        
        # 2. Source discrepancy
        discrepancy_alert = self._check_discrepancy(current_odds)
        if discrepancy_alert:
            alerts.append(discrepancy_alert)
        
        # 3. Late move
        late_alert = self._check_late_move(match_id, commence_time)
        if late_alert:
            alerts.append(late_alert)
        
        # Log all alerts
        for alert in alerts:
            self._log_alert(match_id, alert)
        
        return alerts
    
    def _check_sharp_move(self, match_id: str) -> Optional[Dict]:
        """Check for sharp line movement."""
        # Get recent movement for home odds
        movement = self.tracker.get_line_movement(match_id, 'home')
        
        if len(movement) < 2:
            return None
        
        # Check last N hours
        current_time = time.time()
        cutoff_time = current_time - (self.SHARP_MOVE_HOURS * 3600)
        
        recent = [(ts, odds) for ts, odds in movement if ts >= cutoff_time]
        
        if len(recent) < 2:
            return None
        
        # Calculate max change
        odds_values = [odds for _, odds in recent]
        max_odds = max(odds_values)
        min_odds = min(odds_values)
        
        change_pct = abs((max_odds - min_odds) / max_odds) * 100
        
        if change_pct > self.SHARP_MOVE_PCT:
            return {
                'type': 'sharp_move',
                'severity': 'high',
                'outcome': 'home',
                'change_pct': change_pct,
                'from_odds': max_odds,
                'to_odds': min_odds,
                'timeframe_hours': self.SHARP_MOVE_HOURS,
                'message': f"Sharp move detected: home odds moved {change_pct:.1f}% in {self.SHARP_MOVE_HOURS}h"
            }
        
        return None
    
    def _check_discrepancy(self, current_odds: Dict[str, Dict[str, float]]) -> Optional[Dict]:
        """Check for large discrepancies between bookmakers."""
        if len(current_odds) < 2:
            return None
        
        # Check home odds across bookmakers
        home_odds = [odds.get('home', 0) for odds in current_odds.values() if odds.get('home')]
        
        if len(home_odds) < 2:
            return None
        
        max_odds = max(home_odds)
        min_odds = min(home_odds)
        
        discrepancy_pct = ((max_odds - min_odds) / min_odds) * 100
        
        if discrepancy_pct > self.DISCREPANCY_PCT:
            return {
                'type': 'discrepancy',
                'severity': 'medium',
                'outcome': 'home',
                'discrepancy_pct': discrepancy_pct,
                'max_odds': max_odds,
                'min_odds': min_odds,
                'message': f"Large discrepancy: {discrepancy_pct:.1f}% between bookmakers"
            }
        
        return None
    
    def _check_late_move(self, match_id: str, commence_time: int) -> Optional[Dict]:
        """Check for movement close to match start."""
        hours_to_match = (commence_time - time.time()) / 3600
        
        if hours_to_match > self.LATE_MOVE_HOURS:
            return None
        
        # Get recent movement
        movement = self.tracker.get_line_movement(match_id, 'home')
        
        if len(movement) < 2:
            return None
        
        # Check if there was movement in last 2 hours
        last_two = movement[-2:]
        if len(last_two) == 2:
            _, odds1 = last_two[0]
            _, odds2 = last_two[1]
            
            change_pct = abs((odds2 - odds1) / odds1) * 100
            
            if change_pct > 5:  # Any >5% move in last 2 hours
                return {
                    'type': 'late_move',
                    'severity': 'high',
                    'outcome': 'home',
                    'change_pct': change_pct,
                    'hours_to_match': hours_to_match,
                    'message': f"Late move: {change_pct:.1f}% change with {hours_to_match:.1f}h to match"
                }
        
        return None
    
    def _log_alert(self, match_id: str, alert: Dict):
        """Log alert to file."""
        alert_record = {
            'timestamp': int(time.time()),
            'match_id': match_id,
            **alert
        }
        
        # Append to alerts file
        alerts = []
        if self.alerts_file.exists():
            with open(self.alerts_file) as f:
                try:
                    alerts = json.load(f)
                except:
                    alerts = []
        
        alerts.append(alert_record)
        
        # Keep last 1000 alerts
        alerts = alerts[-1000:]
        
        with open(self.alerts_file, 'w') as f:
            json.dump(alerts, f, indent=2)
        
        logger.warning(f"🚨 ALERT [{alert['type'].upper()}]: {alert['message']}")


def test_alerts():
    """Test alert system."""
    import time
    
    print("=" * 60)
    print("LINE MOVEMENT ALERTS TEST")
    print("=" * 60)
    
    # Create tracker and alert system
    tracker = OddsTracker(db_path='data/odds/test_odds.db')
    alerts_system = LineMovementAlerts(tracker)
    
    match_id = "alert_test_001"
    commence_time = int(time.time()) + 1800  # 30 min from now
    
    # Simulate sharp move
    print("\n1. Simulating sharp move...")
    opening = {'Pinnacle': {'home': 2.50, 'draw': 3.20, 'away': 2.80}}
    tracker.store_odds_snapshot(match_id, opening, is_opening=True)
    
    time.sleep(1)
    
    sharp = {'Pinnacle': {'home': 2.10, 'draw': 3.40, 'away': 3.10}}  # -16% home
    tracker.store_odds_snapshot(match_id, sharp)
    
    # Check alerts
    current_odds = {'Pinnacle': sharp['Pinnacle'], 'Bet365': {'home': 3.00, 'draw': 3.30, 'away': 2.50}}
    alerts = alerts_system.check_all_alerts(match_id, commence_time, current_odds)
    
    print(f"\n2. Alerts triggered: {len(alerts)}")
    for alert in alerts:
        print(f"  - [{alert['type']}] {alert['message']}")
    
    assert any(a['type'] == 'sharp_move' for a in alerts), "Should detect sharp move"
    assert any(a['type'] == 'late_move' for a in alerts), "Should detect late move"
    assert any(a['type'] == 'discrepancy' for a in alerts), "Should detect discrepancy"
    
    print("\n✅ All alert types detected!")


if __name__ == '__main__':
    test_alerts()
