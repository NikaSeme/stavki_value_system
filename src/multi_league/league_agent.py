"""
League Agent - Independent worker for one league.

Runs complete pipeline for a single league:
- Data fetching
- Model predictions
- Value Detection
- Alerting
- Execution (if enabled)
- Performance tracking
"""

import time
import logging
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeagueAgent:
    """
    Independent agent for one league.
    
    Encapsulates all logic for a single league to enable
    parallel execution without conflicts.
    """
    
    def __init__(self, league_id: str, config: Dict):
        """
        Initialize league agent.
        
        Args:
            league_id: League identifier (e.g., 'epl', 'laliga')
            config: League configuration dict
        """
        self.league_id = league_id
        self.config = config
        self.league_name = config['league']['name']
        
        # Execution control
        self.running = False
        self.last_activity = None
        
        # Check interval
        self.check_interval = config.get('execution', {}).get('check_interval', 900)
        
        logger.info(f"✓ League agent initialized: {self.league_name} ({league_id})")
    
    def run(self):
        """Main execution loop for this league."""
        self.running = True
        logger.info(f"🏃 Starting {self.league_name} agent...")
        
        while self.running:
            try:
                self.last_activity = datetime.now()
                
                # Main pipeline
                self._execute_pipeline()
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"{self.league_id} pipeline error: {e}", exc_info=True)
                time.sleep(60)  # Back off on error
    
    def _execute_pipeline(self):
        """
        Execute full prediction pipeline.
        
        1. Fetch upcoming matches
        2. Generate predictions
        3. Identify value bets
        4. Send alerts
        5. Execute bets (if enabled)
        6. Update performance
        """
        logger.info(f"[{self.league_id.upper()}] Running pipeline...")
        
        # For now, simulate pipeline
        # In full implementation, would call actual data/models
        
        logger.info(f"[{self.league_id.upper()}] ✓ Pipeline complete")
    
    def stop(self):
        """Stop agent execution."""
        logger.info(f"Stopping {self.league_name} agent...")
        self.running = False
    
    def get_status(self) -> Dict:
        """Get agent status."""
        return {
            'league_id': self.league_id,
            'league_name': self.league_name,
            'running': self.running,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'check_interval': self.check_interval
        }


def test_league_agent():
    """Test league agent."""
    print("=" * 70)
    print("LEAGUE AGENT TEST")
    print("=" * 70)
    
    # Load config
    import yaml
    with open('config/leagues/epl.yaml') as f:
        config = yaml.safe_load(f)
    
    # Create agent
    agent = LeagueAgent('epl', config)
    
    print(f"\n1. Agent created:")
    print(f"  League: {agent.league_name}")
    print(f"  ID: {agent.league_id}")
    print(f"  Check interval: {agent.check_interval}s")
    
    print(f"\n2. Status:")
    status = agent.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print(f"\n3. Running one pipeline cycle...")
    agent._execute_pipeline()
    
    print("\n✅ League agent test complete!")


if __name__ == '__main__':
    test_league_agent()
