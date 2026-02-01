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
from .league_loader import LeagueModelLoader
from .neural_predictor import NeuralPredictor
from src.strategy.blending import get_internal_weights
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple models.
    
    Supports:
    - 2-model: Poisson + CatBoost 
    - 3-model: Poisson + CatBoost + Neural
    """
    
    def __init__(self, ensemble_file='models/ensemble_simple_latest.pkl', use_neural=False):
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
        self.catboost_loader = None  # New: per-league loader
        self.catboost_legacy = None  # Fallback: old loader
        self.neural = None
        self.calibrators = None
        self.method = None
        self.use_legacy_calibration = True
        # Default Weights (Heuristic) - Safe 2-Model Mix
        self.weights = {'catboost': 0.70, 'neural': 0.0, 'poisson': 0.30}
        
        self._load_ensemble()
        
        # Log Neural status
        if use_neural and not self.use_neural:
            logger.warning("Neural model requested but not found. Falling back to 2-model ensemble.")
        elif not use_neural:
            logger.info("ℹ️ Safe Mode: Neural Model disabled by default.")

    def _load_ensemble(self):
        """Load ensemble model and components."""
        logger.info(f"Loading ensemble from {self.ensemble_file}")
        
        # Load Optimized Weights if available
        meta_path = Path('models') / 'ensemble_optimized_metadata.json'
        if meta_path.exists():
             try:
                 with open(meta_path) as f:
                     meta = json.load(f)
                     if 'weights' in meta:
                         self.weights = meta['weights']
                         self.use_legacy_calibration = False # Disable legacy calibration for optimized weights
                         logger.info(f"✓ Loaded Optimized Weights: {self.weights} (Legacy Calibration Disabled)")
             except Exception as e:
                 logger.warning(f"Failed to load optimized weights: {e}")
        
        # Load ensemble config
        with open(self.ensemble_file, 'rb') as f:
            ensemble_data = pickle.load(f)
        
        self.method = ensemble_data['method']
        self.calibrators = ensemble_data['calibrators']
        
        logger.info(f"Ensemble method: {self.method}")
        
        # Load Poisson model
        from src.models.poisson_model import PoissonMatchPredictor
        poisson_file = Path('models') / 'poisson_v1_latest.pkl'
        self.poisson = PoissonMatchPredictor.load(poisson_file)
        logger.info("✓ Loaded Poisson model (Model A)")
        
        # Load CatBoost per-league models (NEW)
        try:
            self.catboost_loader = LeagueModelLoader()
            logger.info(f"✓ Loaded per-league CatBoost models (Model B) - {len(self.catboost_loader.list_available_models())} sport keys")
        except Exception as e:
            logger.warning(f"Failed to load per-league models: {e}")
            self.catboost_loader = None
        
        # Fallback: legacy CatBoost loader
        if self.catboost_loader is None:
            try:
                self.catboost_legacy = ModelLoader()
                self.catboost_legacy.load_latest()
                logger.info("✓ Loaded legacy CatBoost model (fallback)")
            except Exception as e:
                logger.error(f"Failed to load any CatBoost model: {e}")
        
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
            # Determine sport_key for per-league model routing
            sport_key = None
            if 'sport_key' in events.columns and not events.empty:
                sport_key = events.iloc[0]['sport_key']
            elif 'League' in events.columns and not events.empty:
                # Map league codes to sport keys
                league = events.iloc[0]['League']
                league_to_sport = {
                    'E0': 'soccer_epl', 'SP1': 'soccer_spain_la_liga',
                    'I1': 'soccer_italy_serie_a', 'D1': 'soccer_germany_bundesliga',
                    'F1': 'soccer_france_ligue_one', 'ENG2': 'soccer_england_league1',
                }
                sport_key = league_to_sport.get(league, 'soccer_epl')
            
            # Use per-league loader if available
            if self.catboost_loader is not None:
                catboost_probs = self.catboost_loader.predict(X, sport_key=sport_key or 'soccer_epl')
            elif self.catboost_legacy is not None:
                # Legacy fallback
                X_cb = X.copy()
                expected_cols = self.catboost_legacy.get_feature_names()
                if expected_cols:
                    for col in expected_cols:
                        if col not in X_cb.columns:
                            if col in ['HomeTeam', 'AwayTeam', 'League', 'Season']:
                                X_cb[col] = "Unknown"
                            else:
                                X_cb[col] = 0.0
                    X_cb = X_cb[expected_cols]
                catboost_probs = self.catboost_legacy.predict(X_cb)
            else:
                logger.error("No CatBoost model available!")
                catboost_probs = np.ones((len(events), 3)) / 3  # Uniform fallback
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
                # Filter out the categorical features and extra line movement feats.
                # We select strictly numeric columns that were in the original V1 training set.
                
                # Standard V1 Numeric Features (22 total)
                # Derived from: 3 Elo + 3 HomeForm + 3 AwayForm + 3 HomeOverall + 3 AwayOverall (dup?)
                # Actually, simply filtering numeric columns usually works if Extractor is aligned.
                # But X has 33 features (including line_movement, etc.)
                
                # Filter 1: Exclude obvious non-numeric
                numeric_candidates = X.select_dtypes(include=[np.number]).copy()
                
                # Filter 2: Exclude Line Movement if Neural wasn't trained on it
                # Neural V1 likely wasn't trained on 'sharp_move_detected' etc.
                cols_to_drop = [
                    'sharp_move_detected', 'odds_volatility', 
                    'time_to_match_hours', 'market_efficiency_score'
                ]
                numeric_candidates = numeric_candidates.drop(columns=[c for c in cols_to_drop if c in numeric_candidates.columns])

                # Ensure we have exactly 22 columns (truncate or pad)
                # If we still have > 22, take first 22 (assuming order is preserved from extractor)
                if len(numeric_candidates.columns) > 22:
                     logger.debug(f"Truncating Neural features from {len(numeric_candidates.columns)} to 22")
                     numeric_candidates = numeric_candidates.iloc[:, :22]
                
                if len(numeric_candidates.columns) != 22:
                    logger.warning(f"Neural feature count mismatch: {len(numeric_candidates.columns)} (expected 22)")
                
                neural_probs = self.neural.predict(numeric_candidates.values)
            except Exception as e:
                logger.error(f"Neural prediction failed: {e}")
                neural_probs = None

        # Determine Weights Logic (Dynamic vs Static)
        # Try to get league-specific weights from the first event
        # (Assuming batch consists of one league, which is standard in run_value_finder)
        
        weights = self.weights.copy()
        
        if not events.empty:
            # Check for sport_key or League column
            sport_key = None
            if 'sport_key' in events.columns:
                sport_key = events.iloc[0]['sport_key']
            elif 'League' in events.columns:
                sport_key = events.iloc[0]['League']
            
            if sport_key:
                dynamic_weights = get_internal_weights(sport_key)
                if dynamic_weights:
                     weights = dynamic_weights
                     # logger.info(f"Using Dynamic Weights for {sport_key}: {weights}")

        if neural_probs is not None:
            # 3-Model Ensemble
            w_c = weights['catboost']
            w_n = weights['neural']
            w_p = weights['poisson']
            
            # logger.info(f"Using 3-model Ensemble (Weights: CB={w_c:.2f}, NN={w_n:.2f}, PS={w_p:.2f})")
            
            ensemble_probs_raw = (
                (catboost_probs * w_c) + 
                (neural_probs * w_n) + 
                (poisson_probs * w_p)
            )
        else:
            # 2-Model Fallback (Neural Unavailable)
            w_c_raw = weights['catboost']
            w_p_raw = weights['poisson']
            
            total = w_c_raw + w_p_raw
            if total > 0:
                w_c = w_c_raw / total
                w_p = w_p_raw / total
            else:
                w_c, w_p = 0.7, 0.3 # Fallback
            
            # logger.info(f"Using 2-model Ensemble (Renormalized: CB={w_c:.2f}, PS={w_p:.2f}) - Neural Unavailable")
            ensemble_probs_raw = (catboost_probs * w_c) + (poisson_probs * w_p)
        
        # Apply calibration
        if self.use_legacy_calibration:
            ensemble_probs = self._apply_calibration(ensemble_probs_raw)
        else:
            ensemble_probs = ensemble_probs_raw
        
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
        """
        Apply calibration (supports both Platt and Isotonic).
        
        Platt scaling (LogisticRegression) provides smooth extrapolation
        for out-of-distribution data (e.g., unseen leagues).
        """
        calibrated = np.zeros_like(probs)
        
        for i, cal in enumerate(self.calibrators):
            # Check calibrator type
            if hasattr(cal, 'predict_proba'):
                # Platt scaling (LogisticRegression)
                X = probs[:, i].reshape(-1, 1)
                calibrated[:, i] = cal.predict_proba(X)[:, 1]
            else:
                # Isotonic regression (legacy)
                calibrated[:, i] = cal.predict(probs[:, i])
        
        # Renormalize
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)  # Avoid division by zero
        calibrated = calibrated / row_sums
        
        return calibrated
