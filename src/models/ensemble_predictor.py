"""
Ensemble predictor for live predictions.

Combines Poisson (Model A), CatBoost (Model B), and Neural (Model C) predictions.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import logging

from .loader import ModelLoader
from .neural_predictor import NeuralPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple models.
    
    Supports:
    - 2-model: Poisson + CatBoost 
    - 3-model: Poisson + CatBoost + Neural
    """
    
    def __init__(self, ensemble_file='models/ensemble_simple_latest.pkl', use_neural=True):
        """
        Initialize ensemble predictor.
        
        Args:
            ensemble_file: Path to ensemble model file
            use_neural: Whether to include Model C (neural network)
        """
        self.ensemble_file = Path(ensemble_file)
        # Auto-detect neural model if use_neural is requested
        neural_path = Path('models') / 'neural_v1_latest.pt'
        self.use_neural = use_neural and neural_path.exists()
        
        self.poisson = None
        self.catboost = None
        self.neural = None
        self.calibrators = None
        self.method = None
        
        self._load_ensemble()
        
        # Log Neural status
        if use_neural and not self.use_neural:
            logger.warning("Neural model requested but not found. Falling back to 2-model ensemble.")

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
        
        # Load Neural model if requested
        if self.use_neural:
            try:
                self.neural = NeuralPredictor()
                logger.info("✓ Loaded Neural model (Model C)")
            except Exception as e:
                logger.warning(f"Failed to load neural model: {e}")
                logger.warning("Continuing with 2-model ensemble")
                self.use_neural = False

    def predict(self, events: pd.DataFrame, odds_df: pd.DataFrame, sentiment_data: Dict = None, features: pd.DataFrame = None) -> Tuple[np.ndarray, Dict]:
        """
        Predict probabilities using ensemble.
        
        Args:
            events: DataFrame with event details
            odds_df: Current odds data
            sentiment_data: Optional dictionary of sentiment scores per event
            features: Optional precomputed features (to preserve Elo state)
            
        Returns:
            Tuple of (probabilities array, components dict)
        """
        # Extract features for CatBoost/Neural
        if features is not None:
            X = features
        else:
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
        
        # Get predictions from models
        try:
            poisson_probs = self.poisson.predict(poisson_events)[['prob_home', 'prob_draw', 'prob_away']].values
        except Exception as e:
            logger.error(f"Poisson prediction failed: {e}")
            poisson_probs = np.zeros((len(events), 3))

        try:
            # CatBoost Feature Selection
            # CatBoost Model A (v1) only used ~10 features (FEATURE_COLS).
            # Neural Model B (v1) and Extractor use 22 features.
            # We must filter X for CatBoost to avoid shape mismatch.
            X_cb = X
            if self.catboost:
                 expected_cols = self.catboost.get_feature_names()
                 # If no metadata, attempt fallback to common columns or use all (risky)
                 if not expected_cols:
                     # Fallback: Hardcoded list from ml_model.py v1
                     expected_cols = [
                        'home_goals_for_avg_5', 'home_goals_against_avg_5', 'home_points_avg_5',
                        'home_form_points_5', 'home_matches_count',
                        'away_goals_for_avg_5', 'away_goals_against_avg_5', 'away_points_avg_5',
                        'away_form_points_5', 'away_matches_count'
                     ]
                 
                 # Ensure columns exist (fill 0 if missing)
                 X_cb_df = X.copy()
                 for col in expected_cols:
                     if col not in X_cb_df.columns:
                         X_cb_df[col] = 0.0
                 X_cb = X_cb_df[expected_cols]

            catboost_probs = self.catboost.predict(X_cb)
        except Exception as e:
            logger.error(f"CatBoost prediction failed: {e}")
            catboost_probs = np.zeros((len(events), 3))
        
        # Combine based on available models using "Privilege" weights
        # Hierarchy: CatBoost (50%) > Neural (30%) > Poisson (20%)
        
        neural_probs = None
        # Try to get Neural predictions first
        if self.use_neural and self.neural:
            try:
                # Neural expects exactly 22 numeric features.
                # Filter out the categorical features we added for CatBoost.
                numeric_cols = [c for c in X.columns if c not in ['HomeTeam', 'AwayTeam', 'Season']]
                X_neural = X[numeric_cols]
                
                # Ensure we have exactly 22 columns (fill/truncate if needed for safety)
                if len(X_neural.columns) != 22:
                    logger.warning(f"Neural feature count mismatch: {len(X_neural.columns)} (expected 22)")
                
                neural_probs = self.neural.predict(X_neural.values)
            except Exception as e:
                logger.error(f"Neural prediction failed: {e}")
                neural_probs = None

        if neural_probs is not None:
            # 3-Model Ensemble (Preferred)
            # Weights: Valued based on "Brainstorming" analysis
            # CatBoost (0.5): The robust generalist
            # Neural (0.3): The non-linear specialist
            # Poisson (0.2): The statistical anchor
            ensemble_probs_raw = (
                (catboost_probs * 0.50) + 
                (neural_probs * 0.30) + 
                (poisson_probs * 0.20)
            )
            logger.info("Using 3-model Weighted Ensemble (CB:0.5, NN:0.3, PS:0.2)")
        else:
            # 2-Model Fallback
            # Weights: CatBoost (0.7) > Poisson (0.3)
            ensemble_probs_raw = (catboost_probs * 0.70) + (poisson_probs * 0.30)
            logger.info("Using 2-model Weighted Ensemble (CB:0.7, PS:0.3) - Neural Unavailable")
        
        # Apply calibration
        ensemble_probs = self._apply_calibration(ensemble_probs_raw)
        
        # Store components for logging
        components = {
            'poisson': poisson_probs,
            'catboost': catboost_probs,
            'ensemble_raw': ensemble_probs_raw,
        }
        
        if neural_probs is not None:
            components['neural'] = neural_probs
            
        # Add sentiment if provided (for downstream decision logic)
        if sentiment_data:
            components['sentiment'] = sentiment_data
        
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
