"""
Per-League Ensemble Predictor (V2)

Clean architecture combining:
- Model A: Poisson (per-league baselines)
- Model B: CatBoost (6 per-league models)
- Meta: Configurable per-league weights

Replaces legacy ensemble_predictor.py
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

from .poisson_model import PoissonMatchPredictor
from .league_loader import LeagueModelLoader
from .feature_contract import load_contract

logger = logging.getLogger(__name__)

# Default weights per league (fallback if no config)
DEFAULT_LEAGUE_WEIGHTS = {
    'soccer_epl': {'catboost': 0.457, 'neural': 0.23, 'poisson': 0.313},
    'soccer_spain_la_liga': {'catboost': 0.636, 'neural': 0.032, 'poisson': 0.333},
    'soccer_italy_serie_a': {'catboost': 0.47, 'neural': 0.387, 'poisson': 0.143},
    'soccer_germany_bundesliga': {'catboost': 0.708, 'neural': 0.207, 'poisson': 0.085},
    'soccer_france_ligue_one': {'catboost': 0.416, 'neural': 0.182, 'poisson': 0.402},
    'soccer_efl_champ': {'catboost': 0.391, 'neural': 0.376, 'poisson': 0.233},
    'soccer_england_league1': {'catboost': 0.50, 'neural': 0.20, 'poisson': 0.30},
    '_default': {'catboost': 0.50, 'neural': 0.20, 'poisson': 0.30},
}

# Team name normalization
TEAM_ALIASES = {
    # EPL
    'man utd': 'Man United',
    'man city': 'Man City',
    'spurs': 'Tottenham',
    'wolves': 'Wolverhampton',
    # Bundesliga
    'bayern': 'Bayern Munich',
    'dortmund': 'Dortmund',
    'leverkusen': 'Bayer Leverkusen',
    'gladbach': 'M\'gladbach',
    # Ligue 1
    'psg': 'Paris SG',
    'om': 'Marseille',
    'ol': 'Lyon',
    # La Liga
    'real': 'Real Madrid',
    'barca': 'Barcelona',
    'atleti': 'Ath Madrid',
    # Serie A
    'inter': 'Inter',
    'juve': 'Juventus',
    'ac milan': 'Milan',
}


def normalize_team_name(name: str) -> str:
    """Normalize team name for consistent matching."""
    if not name:
        return name
    lower = name.lower().strip()
    return TEAM_ALIASES.get(lower, name)


class PerLeagueEnsemble:
    """
    Clean ensemble predictor using per-league models.
    
    Architecture:
        - Model A (Poisson): Historical team strengths, good for new teams
        - Model B (CatBoost): ML with odds/form features, per-league specialized
        - Blend: Weighted average per league
    
    Usage:
        ensemble = PerLeagueEnsemble()
        probs, info = ensemble.predict(events_df, features_df)
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize ensemble with models and weights.
        
        Args:
            config_path: Optional path to league_config.json for custom weights
        """
        self.models_dir = Path(__file__).parent.parent.parent / 'models'
        self.contract = load_contract()
        
        # Load per-league weights
        self.weights = self._load_weights(config_path)
        
        # Initialize models
        self.poisson = self._load_poisson()
        self.catboost_loader = self._load_catboost()
        
        logger.info(f"PerLeagueEnsemble initialized: {len(self.weights)} league configs")
    
    def _load_weights(self, config_path: Optional[Path]) -> Dict[str, Dict[str, float]]:
        """Load per-league weights from config or use defaults."""
        weights = DEFAULT_LEAGUE_WEIGHTS.copy()
        
        # Try to load from config file
        if config_path is None:
            config_path = self.models_dir / 'league_weights.json'
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    custom = json.load(f)
                
                # Handle nested format: {"soccer_epl": {"weights": {...}, "alpha": ...}}
                for sport_key, config in custom.items():
                    if isinstance(config, dict):
                        if 'weights' in config:
                            # Nested format from league_weights.json
                            weights[sport_key] = config['weights']
                        else:
                            # Flat format: {"soccer_epl": {"catboost": 0.5, ...}}
                            weights[sport_key] = config
                
                logger.info(f"Loaded custom weights from {config_path} ({len(custom)} leagues)")
            except Exception as e:
                logger.warning(f"Failed to load weights: {e}, using defaults")
        
        return weights
    
    def _load_poisson(self) -> Optional[PoissonMatchPredictor]:
        """Load Poisson model."""
        poisson_path = self.models_dir / 'poisson_v1_latest.pkl'
        
        if not poisson_path.exists():
            logger.warning("Poisson model not found, predictions will be CatBoost-only")
            return None
        
        try:
            model = PoissonMatchPredictor.load(poisson_path)
            logger.info(f"✓ Loaded Poisson model ({len(model.team_attack)} teams)")
            return model
        except Exception as e:
            logger.error(f"Failed to load Poisson: {e}")
            return None
    
    def _load_catboost(self) -> Optional[LeagueModelLoader]:
        """Load per-league CatBoost models."""
        try:
            loader = LeagueModelLoader(self.models_dir)
            logger.info(f"✓ Loaded CatBoost loader ({len(loader.list_available_models())} sport keys)")
            return loader
        except Exception as e:
            logger.error(f"Failed to load CatBoost loader: {e}")
            return None
    
    def get_weights(self, sport_key: str) -> Dict[str, float]:
        """Get blend weights for a sport key."""
        if sport_key in self.weights:
            w = self.weights[sport_key]
            # Handle nested format
            if isinstance(w, dict) and 'weights' in w:
                return w['weights']
            return w
        return self.weights.get('_default', {'catboost': 0.5, 'neural': 0.2, 'poisson': 0.3})
    
    def _predict_poisson(self, events: pd.DataFrame) -> np.ndarray:
        """Get Poisson predictions."""
        if self.poisson is None:
            return None
        
        # Prepare events for Poisson (needs HomeTeam, AwayTeam)
        poisson_df = events.copy()
        
        # Normalize column names
        if 'home_team' in poisson_df.columns:
            poisson_df = poisson_df.rename(columns={
                'home_team': 'HomeTeam',
                'away_team': 'AwayTeam'
            })
        
        # Normalize team names
        if 'HomeTeam' in poisson_df.columns:
            poisson_df['HomeTeam'] = poisson_df['HomeTeam'].apply(normalize_team_name)
            poisson_df['AwayTeam'] = poisson_df['AwayTeam'].apply(normalize_team_name)
        
        try:
            preds = self.poisson.predict(poisson_df)
            probs = preds[['prob_home', 'prob_draw', 'prob_away']].values
            return probs
        except Exception as e:
            logger.error(f"Poisson prediction failed: {e}")
            return None
    
    def _predict_catboost(self, features: pd.DataFrame, sport_key: str) -> np.ndarray:
        """Get CatBoost predictions for a league."""
        if self.catboost_loader is None:
            return None
        
        try:
            probs = self.catboost_loader.predict(features, sport_key=sport_key)
            return probs
        except Exception as e:
            logger.error(f"CatBoost prediction failed for {sport_key}: {e}")
            return None
    
    def predict(
        self,
        events: pd.DataFrame,
        features: pd.DataFrame,
        sport_key: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict match probabilities using ensemble.
        
        Args:
            events: DataFrame with event info (home_team, away_team, etc.)
            features: DataFrame with ML features (28 columns per contract)
            sport_key: API sport key for model routing (e.g., 'soccer_epl')
        
        Returns:
            Tuple of (probabilities [N, 3], info dict with components)
        """
        n_samples = len(events)
        
        # Determine sport_key if not provided
        if sport_key is None:
            if 'sport_key' in events.columns and not events.empty:
                sport_key = events.iloc[0]['sport_key']
            else:
                sport_key = 'soccer_epl'  # Default
        
        # Get predictions from each model
        poisson_probs = self._predict_poisson(events)
        catboost_probs = self._predict_catboost(features, sport_key)
        
        # Get weights for this league
        weights = self.get_weights(sport_key)
        w_cb = weights.get('catboost', 0.6)
        w_ps = weights.get('poisson', 0.4)
        
        # Blend predictions
        if catboost_probs is not None and poisson_probs is not None:
            # Both available: weighted blend
            ensemble_probs = (w_cb * catboost_probs) + (w_ps * poisson_probs)
        elif catboost_probs is not None:
            # CatBoost only
            ensemble_probs = catboost_probs
            logger.warning("Using CatBoost-only (Poisson unavailable)")
        elif poisson_probs is not None:
            # Poisson only
            ensemble_probs = poisson_probs
            logger.warning("Using Poisson-only (CatBoost unavailable)")
        else:
            # No models available - uniform distribution
            logger.error("No models available! Returning uniform probabilities")
            ensemble_probs = np.ones((n_samples, 3)) / 3
        
        # Ensure probabilities sum to 1
        row_sums = ensemble_probs.sum(axis=1, keepdims=True)
        ensemble_probs = ensemble_probs / np.maximum(row_sums, 1e-10)
        
        # Build info dict
        info = {
            'sport_key': sport_key,
            'weights': weights,
            'poisson': poisson_probs,
            'catboost': catboost_probs,
            'ensemble': ensemble_probs,
        }
        
        return ensemble_probs, info
    
    def predict_batch(
        self,
        events_by_league: Dict[str, pd.DataFrame],
        features_by_league: Dict[str, pd.DataFrame]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict for multiple leagues at once.
        
        Args:
            events_by_league: Dict mapping sport_key -> events DataFrame
            features_by_league: Dict mapping sport_key -> features DataFrame
        
        Returns:
            Concatenated probabilities and info per league
        """
        all_probs = []
        all_info = {}
        
        for sport_key in events_by_league:
            events = events_by_league[sport_key]
            features = features_by_league[sport_key]
            
            probs, info = self.predict(events, features, sport_key)
            all_probs.append(probs)
            all_info[sport_key] = info
        
        return np.vstack(all_probs), all_info


def test_ensemble():
    """Test the per-league ensemble."""
    print("\n" + "=" * 60)
    print("PER-LEAGUE ENSEMBLE TEST")
    print("=" * 60)
    
    # Initialize
    ensemble = PerLeagueEnsemble()
    
    # Test data
    events = pd.DataFrame([{
        'event_id': 'test_1',
        'home_team': 'Arsenal',
        'away_team': 'Chelsea',
        'sport_key': 'soccer_epl',
    }])
    
    features = pd.DataFrame([{
        'odds_home': 2.10, 'odds_draw': 3.40, 'odds_away': 3.50,
        'implied_home': 0.476, 'implied_draw': 0.294, 'implied_away': 0.286,
        'no_vig_home': 0.45, 'no_vig_draw': 0.28, 'no_vig_away': 0.27,
        'market_overround': 1.056,
        'line_dispersion_home': 0.02, 'line_dispersion_draw': 0.02, 'line_dispersion_away': 0.02,
        'book_count': 5.0,
        'elo_home': 1650.0, 'elo_away': 1600.0, 'elo_diff': 50.0,
        'form_pts_home_l5': 10.0, 'form_pts_away_l5': 8.0,
        'form_gf_home_l5': 9.0, 'form_gf_away_l5': 7.0,
        'form_ga_home_l5': 4.0, 'form_ga_away_l5': 5.0,
        'rest_days_home': 7.0, 'rest_days_away': 7.0,
        'league': 'epl', 'home_team': 'Arsenal', 'away_team': 'Chelsea',
    }])
    
    # Predict
    probs, info = ensemble.predict(events, features)
    
    print(f"\n📊 Prediction for Arsenal vs Chelsea (EPL):")
    print(f"   Weights: {info['weights']}")
    print(f"   Poisson:  H={info['poisson'][0,0]:.3f}, D={info['poisson'][0,1]:.3f}, A={info['poisson'][0,2]:.3f}")
    print(f"   CatBoost: H={info['catboost'][0,0]:.3f}, D={info['catboost'][0,1]:.3f}, A={info['catboost'][0,2]:.3f}")
    print(f"   Ensemble: H={probs[0,0]:.3f}, D={probs[0,1]:.3f}, A={probs[0,2]:.3f}")
    print(f"   Sum: {probs[0].sum():.4f}")
    
    print("\n✅ Test passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_ensemble()
