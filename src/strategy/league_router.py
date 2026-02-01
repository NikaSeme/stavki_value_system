
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class LeagueRouter:
    """
    Decides betting strategy and model weights based on the League.
    driven by models/league_config.json.
    """
    
    def __init__(self, config_path=None):
        if config_path is None:
            # Default location
            config_path = Path(__file__).parent.parent.parent / 'models' / 'league_config.json'
            
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.league_map = self._normalize_league_names()
        
    def _load_config(self):
        if not self.config_path.exists():
            logger.warning(f"Config config not found at {self.config_path}. Using safe defaults.")
            return {"default": {"policy": "SKIP", "weights": {"catboost": 0, "neural": 0, "poisson": 0}}, "leagues": {}}
            
        with open(self.config_path, 'r') as f:
            return json.load(f)
            
    def _normalize_league_names(self):
        # Helper to map common names to config keys
        # The keys in config are typically lowercase slugs
        return {
            'epl': 'epl', 'premier_league': 'epl', 'premier-league': 'epl',
            'laliga': 'laliga', 'la_liga': 'laliga', 'primera_division': 'laliga',
            'bundesliga': 'bundesliga', 
            'seriea': 'seriea', 'serie_a': 'seriea',
            'ligue1': 'ligue1', 'ligue_1': 'ligue1',
            'championship': 'championship'
        }

    def get_strategy(self, league_name):
        """
        Get policy and weights for a league.
        Returns: (policy_name, weights_dict)
        """
        if not league_name:
            return self.config['default']['policy'], self.config['default']['weights']
            
        slug = self.league_map.get(str(league_name).lower().replace(' ', ''), str(league_name).lower())
        
        league_conf = self.config['leagues'].get(slug)
        
        if league_conf:
            return league_conf['policy'], league_conf['weights']
        else:
            logger.debug(f"Unknown league '{league_name}' (slug: {slug}) - Using DEFAULT")
            return self.config['default']['policy'], self.config['default']['weights']

    def get_weights(self, league_name):
        """Directly get (w_c, w_n, w_p) tuple."""
        _, w = self.get_strategy(league_name)
        return (w.get('catboost', 0.0), w.get('neural', 0.0), w.get('poisson', 0.0))
