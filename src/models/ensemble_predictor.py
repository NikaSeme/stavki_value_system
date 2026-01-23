"""
Ensemble predictor for live predictions.

Combines Poisson (Model A) and CatBoost (Model B) predictions.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import logging

from .loader import ModelLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple models.
    
    Currently supports Poisson + CatBoost averaging with calibration.
    """
    
    def __init__(self, ensemble_file='models/ensemble_simple_latest.pkl'):
        """
        Initialize ensemble predictor.
        
        Args:
            ensemble_file: Path to ensemble model file
        """
        self.ensemble_file = Path(ensemble_file)
        self.poisson = None
        self.catboost = None
        self.calibrators = None
        self.method = None
        
        self._load_ensemble()
    
    def _load_ensemble(self):
        """Load ensemble model and components."""
        logger.info(f"Loading ensemble from {self.ensemble_file}")
        
        # Load ensemble config
        with open(self.ensemble_file, 'rb') as f:
            ensemble_data = pickle.load(f)
        
        self.method = ensemble_data['method']
        self.calibrators = ensemble_data['calibrators']
        
        logger.info(f"Ensemble method: {self.method}")
        
        # Load Poisson model
        from scripts.train_poisson_model import PoissonMatchPredictor
        poisson_file = Path('models') / 'poisson_v1_latest.pkl'
        self.poisson = PoissonMatchPredictor.load(poisson_file)
        logger.info("✓ Loaded Poisson model")
        
        # Load CatBoost model
        self.catboost = ModelLoader()
        self.catboost.load_latest()
        logger.info("✓ Loaded CatBoost model")
    
    def predict(self, events: pd.DataFrame, odds_df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Predict probabilities using ensemble.
        
        Args:
            events: DataFrame with event details
            odds_df: Current odds data
            
        Returns:
            Tuple of (probabilities array, components dict)
        """
        # Extract features for CatBoost
        from src.features.live_extractor import LiveFeatureExtractor
        extractor = LiveFeatureExtractor()
        X = extractor.extract_features(events, odds_df)
        
        # Prepare for Poisson (needs capitalized columns)
        poisson_events = events.copy()
        if 'home_team' in poisson_events.columns:
            poisson_events = poisson_events.rename(columns={
                'home_team': 'HomeTeam',
                'away_team': 'AwayTeam'
            })
        
        # Get predictions from both models
        poisson_probs = self.poisson.predict(poisson_events)[['prob_home', 'prob_draw', 'prob_away']].values
        catboost_probs = self.catboost.predict(X.values)
        
        # Combine based on method
        if self.method == 'simple_average':
            ensemble_probs_raw = (poisson_probs + catboost_probs) / 2.0
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
        
        # Apply calibration
        ensemble_probs = self._apply_calibration(ensemble_probs_raw)
        
        # Store components for logging
        components = {
            'poisson': poisson_probs,
            'catboost': catboost_probs,
            'ensemble_raw': ensemble_probs_raw,
        }
        
        return ensemble_probs, components
    
    def _apply_calibration(self, probs):
        """Apply isotonic calibration."""
        calibrated = np.zeros_like(probs)
        
        for i, cal in enumerate(self.calibrators):
            calibrated[:, i] = cal.predict(probs[:, i])
        
        # Renormalize
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / row_sums
        
        return calibrated
