"""
Multi-League Manager - Orchestrate multiple leagues concurrently.

Coordinates:
- League-specific agents
- Parallel execution
- Unified monitoring
- Cross-league aggregation
"""

import logging
import threading
from typing import Dict, List, Optional
from pathlib import Path
import yaml
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeagueManager:
    """
    Manage multiple leagues/sports concurrently.
    
    Each league runs independently with:
    - Own data sources
    - Own models
    - Own performance tracking
    - Unified alerting
    """
    
    def __init__(self, league_ids: List[str], config_dir='config/leagues'):
        """
        Initialize multi-league manager.
        
        Args:
            league_ids: List of league IDs to manage (e.g., ['epl', 'laliga'])
            config_dir: Directory containing league configs
        """
        self.league_ids = league_ids
        self.config_dir = Path(config_dir)
        
        # Load configurations
        self.configs = {}
        for league_id in league_ids:
            self.configs[league_id] = self.load_league_config(league_id)
        
        # Initialize agents (lazy - created on run)
        self.agents = {}
        
        # Execution control
        self.running = False
        self.threads = {}
        
        logger.info(f"✓ League manager initialized ({len(league_ids)} leagues)")
    
    def load_league_config(self, league_id: str) -> Dict:
        """Load configuration for a league."""
        config_path = self.config_dir / f"{league_id}.yaml"
        
        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}, using defaults")
            return self._get_default_config(league_id)
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded config for {league_id}: {config['league']['name']}")
        return config
    
    def _get_default_config(self, league_id: str) -> Dict:
        """Generate default configuration."""
        return {
            'league': {
                'id': league_id,
                'name': league_id.upper(),
                'sport': 'football',
                'enabled': True
            },
            'models': {
                'enabled': True,
                'model_dir': f'models/{league_id}'
            },
            'execution': {
                'check_interval': 900,  # 15 minutes
                'auto_bet': False,
                'min_ev_threshold': 5.0
            }
        }
    
    def get_active_leagues(self) -> List[str]:
        """Get list of enabled leagues."""
        return [
            league_id for league_id, config in self.configs.items()
            if config['league'].get('enabled', True)
        ]
    
    def start_all_leagues(self):
        """Start all leagues in parallel threads."""
        self.running = True
        active_leagues = self.get_active_leagues()
        
        logger.info(f"Starting {len(active_leagues)} leagues...")
        
        for league_id in active_leagues:
            thread = threading.Thread(
                target=self._run_league,
                args=(league_id,),
                name=f"{league_id}_thread",
                daemon=True
            )
            thread.start()
            self.threads[league_id] = thread
            
            logger.info(f"  ✓ Started {league_id} (thread: {thread.name})")
        
        logger.info("✅ All leagues started")
    
    def _run_league(self, league_id: str):
        """Run individual league (called in thread)."""
        # Import here to avoid circular imports
        from .league_agent import LeagueAgent
        
        try:
            config = self.configs[league_id]
            agent = LeagueAgent(league_id, config)
            self.agents[league_id] = agent
            
            # Run agent's main loop
            agent.run()
            
        except Exception as e:
            logger.error(f"League {league_id} crashed: {e}", exc_info=True)
    
    def stop_all_leagues(self):
        """Stop all running leagues."""
        logger.info("Stopping all leagues...")
        self.running = False
        
        # Signal agents to stop
        for league_id, agent in self.agents.items():
            if hasattr(agent, 'stop'):
                agent.stop()
        
        # Wait for threads
        for league_id, thread in self.threads.items():
            if thread.is_alive():
                thread.join(timeout=5.0)
                logger.info(f"  ✓ Stopped {league_id}")
        
        logger.info("✅ All leagues stopped")
    
    def get_status(self) -> Dict:
        """Get status of all leagues."""
        status = {
            'running': self.running,
            'leagues': {}
        }
        
        for league_id in self.league_ids:
            thread = self.threads.get(league_id)
            agent = self.agents.get(league_id)
            
            status['leagues'][league_id] = {
                'config': self.configs.get(league_id, {}).get('league', {}),
                'thread_alive': thread.is_alive() if thread else False,
                'agent_loaded': agent is not None,
                'last_activity': getattr(agent, 'last_activity', None) if agent else None
            }
        
        return status
    
    def run_forever(self):
        """Start leagues and keep main thread alive."""
        self.start_all_leagues()
        
        try:
            # Keep alive while leagues run
            while self.running:
                time.sleep(10)
                
                # Health check
                for league_id, thread in self.threads.items():
                    if not thread.is_alive() and self.running:
                        logger.warning(f"{league_id} thread died, restarting...")
                        # Could implement auto-restart here
        
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        
        finally:
            self.stop_all_leagues()


def test_league_manager():
    """Test multi-league manager."""
    print("=" * 70)
    print("MULTI-LEAGUE MANAGER TEST")
    print("=" * 70)
    
    # Create manager for 3 leagues
    manager = LeagueManager(['epl', 'laliga', 'bundesliga'])
    
    print("\n1. Loaded Configurations:")
    for league_id, config in manager.configs.items():
        league_info = config['league']
        print(f"  - {league_id}: {league_info['name']} ({league_info['sport']})")
    
    print("\n2. Active Leagues:")
    active = manager.get_active_leagues()
    print(f"  {len(active)} leagues enabled: {', '.join(active)}")
    
    print("\n3. Status:")
    status = manager.get_status()
    print(f"  Running: {status['running']}")
    print(f"  Configured: {len(status['leagues'])} leagues")
    
    print("\n✅ Multi-league manager test complete!")
    print("\nTo start leagues:")
    print("  manager.start_all_leagues()")
    print("  manager.run_forever()  # Blocks")


if __name__ == '__main__':
    test_league_manager()
