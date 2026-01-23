"""
Alert Manager - Central alerting system.

Coordinates all alert types:
- Value bet opportunities
- Line movements
- Performance warnings  
- System health issues
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from .telegram_bot import TelegramAlertBot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertManager:
    """Manage and send all system alerts."""
    
    def __init__(self, config_path='config/alerts_config.json'):
        """Initialize alert manager."""
        self.config = self.load_config(config_path)
        
        # Initialize Telegram bot if enabled
        self.telegram_bot = None
        if self.config.get('telegram', {}).get('enabled', True):
            try:
                self.telegram_bot = TelegramAlertBot()
                logger.info("✓ Telegram bot initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}")
        
        # Alert history for deduplication
        self.alert_history_file = Path('data/logs/alert_history.json')
        self.alert_history_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.alert_history = self.load_alert_history()
    
    def load_config(self, config_path):
        """Load alert configuration."""
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        
        # Default config
        return {
            'telegram': {'enabled': True},
            'thresholds': {
                'min_ev_percent': 5.0,
                'max_drawdown_percent': 20.0,
                'sharp_move_pct': 10.0
            },
            'value_bets': {'enabled': True},
            'line_movements': {'enabled': True},
            'performance': {'enabled': True}
        }
    
    def load_alert_history(self) -> List[Dict]:
        """Load recent alert history."""
        if self.alert_history_file.exists():
            try:
                with open(self.alert_history_file) as f:
                    history = json.load(f)
                    # Keep only last 24 hours
                    cutoff = time.time() - 86400
                    return [a for a in history if a.get('timestamp', 0) > cutoff]
            except:
                return []
        return []
    
    def save_alert_history(self):
        """Save alert history."""
        with open(self.alert_history_file, 'w') as f:
            json.dump(self.alert_history, f, indent=2)
    
    def is_duplicate(self, alert_type: str, key: str, window_minutes: int = 60) -> bool:
        """Check if alert was recently sent."""
        cutoff = time.time() - (window_minutes * 60)
        
        for alert in self.alert_history:
            if (alert['type'] == alert_type and 
                alert['key'] == key and 
                alert['timestamp'] > cutoff):
                return True
        
        return False
    
    def log_alert(self, alert_type: str, key: str, message: str):
        """Log sent alert."""
        self.alert_history.append({
            'type': alert_type,
            'key': key,
            'message': message,
            'timestamp': time.time()
        })
        
        # Keep only last 1000 alerts
        self.alert_history = self.alert_history[-1000:]
        self.save_alert_history()
    
    def send_alert(self, message: str, alert_type: str, key: str) -> bool:
        """
        Send alert via configured channels.
        
        Args:
            message: Alert text
            alert_type: Type of alert
            key: Unique key for deduplication
            
        Returns:
            True if sent successfully
        """
        # Check for duplicates
        if self.is_duplicate(alert_type, key):
            logger.info(f"Skipping duplicate alert: {alert_type}/{key}")
            return False
        
        # Send via Telegram
        if self.telegram_bot:
            success = self.telegram_bot.send_alert_sync(message)
            if success:
                self.log_alert(alert_type, key, message)
                return True
        
        # Fallback: log to file
        logger.warning(f"Alert not sent (no bot): {alert_type}")
        return False
    
    def send_value_bet_alert(self, bet_info: Dict) -> bool:
        """
        Send value bet opportunity alert.
        
        Args:
            bet_info: Dict with match, market, odds, model_prob, ev, stake
        """
        if not self.config.get('value_bets', {}).get('enabled', True):
            return False
        
        # Check EV threshold
        min_ev = self.config.get('thresholds', {}).get('min_ev_percent', 5.0)
        if bet_info.get('ev', 0) < min_ev:
            return False
        
        # Format message
        message = f"""
🎯 **VALUE BET DETECTED**

**Match:** {bet_info['match']}
**Market:** {bet_info['market']}
**Current Odds:** {bet_info['odds']:.2f}
**Model Prob:** {bet_info['model_prob']:.1%}
**EV:** +{bet_info['ev']:.1f}%
**Recommended Stake:** £{bet_info['stake']:.2f}

⚡ Act fast - odds may change
        """.strip()
        
        key = f"{bet_info['match']}_{bet_info['market']}"
        return self.send_alert(message, 'value_bet', key)
    
    def send_line_move_alert(self, movement: Dict) -> bool:
        """
        Send line movement alert.
        
        Args:
            movement: Dict with match, outcome, from_odds, to_odds, change_pct
        """
        if not self.config.get('line_movements', {}).get('enabled', True):
            return False
        
        # Check threshold
        min_change = self.config.get('thresholds', {}).get('sharp_move_pct', 10.0)
        if abs(movement.get('change_pct', 0)) < min_change:
            return False
        
        # Format message
        direction = "📉" if movement['change_pct'] < 0 else "📈"
        message = f"""
{direction} **SHARP LINE MOVE**

**Match:** {movement['match']}
**Outcome:** {movement['outcome'].title()}
**Movement:** {movement['from_odds']:.2f} → {movement['to_odds']:.2f} ({movement['change_pct']:+.1f}%)
**Timeframe:** {movement.get('hours', 'N/A')} hours

🔍 Investigate before betting
        """.strip()
        
        key = f"{movement['match']}_{movement['outcome']}_line"
        return self.send_alert(message, 'line_move', key)
    
    def send_performance_alert(self, perf_data: Dict) -> bool:
        """
        Send performance warning alert.
        
        Args:
            perf_data: Dict with drawdown, roi, clv, recent_record
        """
        if not self.config.get('performance', {}).get('enabled', True):
            return False
        
        # Check if drawdown exceeds threshold
        max_drawdown = self.config.get('thresholds', {}).get('max_drawdown_percent', 20.0)
        current_drawdown = perf_data.get('drawdown', 0)
        
        if current_drawdown < max_drawdown:
            return False
        
        # Format message
        message = f"""
⚠️ **PERFORMANCE WARNING**

**Current Drawdown:** {current_drawdown:.1f}%
**Threshold:** {max_drawdown:.1f}%

**Recent Performance:**
- Last 10 bets: {perf_data.get('recent_record', 'N/A')}
- Week ROI: {perf_data.get('week_roi', 0):+.1f}%
- Week CLV: {perf_data.get('week_clv', 0):+.1f}%

📊 Consider reviewing strategy
        """.strip()
        
        key = "performance_drawdown"
        return self.send_alert(message, 'performance', key)
    
    def send_system_health_alert(self, error_info: Dict) -> bool:
        """
        Send system health alert.
        
        Args:
            error_info: Dict with component, error, last_success
        """
        # Format message
        message = f"""
🔧 **SYSTEM ERROR**

**Component:** {error_info['component']}
**Error:** {error_info['error']}
**Last Success:** {error_info.get('last_success', 'Unknown')}

⚡ Action required
        """.strip()
        
        key = f"system_{error_info['component']}"
        return self.send_alert(message, 'system_health', key)


def test_alerts():
    """Test all alert types."""
    print("=" * 70)
    print("ALERT MANAGER TEST")
    print("=" * 70)
    
    manager = AlertManager()
    
    # Test value bet alert
    print("\n1. Testing value bet alert...")
    bet_info = {
        'match': 'Man City vs Liverpool',
        'market': 'Home Win',
        'odds': 2.15,
        'model_prob': 0.523,
        'ev': 6.7,
        'stake': 50.0
    }
    sent = manager.send_value_bet_alert(bet_info)
    print(f"  {'✓' if sent else '✗'} Value bet alert")
    
    # Test line move alert
    print("\n2. Testing line move alert...")
    movement = {
        'match': 'Chelsea vs Arsenal',
        'outcome': 'home',
        'from_odds': 2.50,
        'to_odds': 2.10,
        'change_pct': -16.0,
        'hours': 6
    }
    sent = manager.send_line_move_alert(movement)
    print(f"  {'✓' if sent else '✗'} Line move alert")
    
    # Test performance alert
    print("\n3. Testing performance alert...")
    perf_data = {
        'drawdown': 22.5,
        'recent_record': '3W-7L',
        'week_roi': -5.2,
       'week_clv': -1.8
    }
    sent = manager.send_performance_alert(perf_data)
    print(f"  {'✓' if sent else '✗'} Performance alert")
    
    # Test system health alert
    print("\n4. Testing system health alert...")
    error_info = {
        'component': 'Odds API',
        'error': 'Connection timeout',
        'last_success': '45 minutes ago'
    }
    sent = manager.send_system_health_alert(error_info)
    print(f"  {'✓' if sent else '✗'} System health alert")
    
    print("\n" + "=" * 70)
    print("✅ Alert tests complete - check Telegram!")
    print("=" * 70)


if __name__ == '__main__':
    test_alerts()
