"""
League Model Loader

Routes predictions to the correct per-league CatBoost model.
Falls back to global V2 model if league not found.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.models.feature_contract import load_contract

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# Mapping from API sport keys to model file names
SPORT_KEY_TO_MODEL = {
    'soccer_epl': 'catboost_epl.cbm',
    'soccer_england_efl_cup': 'catboost_epl.cbm',  # Use EPL for EFL Cup
    'soccer_spain_la_liga': 'catboost_laliga.cbm',
    'soccer_italy_serie_a': 'catboost_seriea.cbm',
    'soccer_germany_bundesliga': 'catboost_bundesliga.cbm',
    'soccer_france_ligue_one': 'catboost_ligue1.cbm',
    'soccer_england_league1': 'catboost_championship.cbm',  # Lower leagues use championship
    'soccer_england_league2': 'catboost_championship.cbm',
}


class LeagueModelLoader:
    """
    Loads and manages per-league CatBoost models.
    
    Usage:
        loader = LeagueModelLoader()
        probs = loader.predict(df, sport_key='soccer_epl')
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        """Initialize loader with models directory."""
        self.models_dir = models_dir or MODELS_DIR
        self.models: Dict[str, CatBoostClassifier] = {}
        self.fallback_model: Optional[CatBoostClassifier] = None
        self.contract = load_contract()
        self.metadata: Dict[str, Any] = {}
        
        # Load metadata if available
        meta_path = self.models_dir / "per_league_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)
        
        logger.info(f"LeagueModelLoader initialized with {len(SPORT_KEY_TO_MODEL)} sport key mappings")
    
    def _load_model(self, model_file: str) -> Optional[CatBoostClassifier]:
        """Load a CatBoost model from file."""
        model_path = self.models_dir / model_file
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return None
        
        try:
            model = CatBoostClassifier()
            model.load_model(str(model_path))
            logger.debug(f"Loaded model: {model_file}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_file}: {e}")
            return None
    
    def get_model(self, sport_key: str) -> CatBoostClassifier:
        """
        Get the appropriate model for a sport key.
        
        Args:
            sport_key: API sport key (e.g., 'soccer_epl')
        
        Returns:
            CatBoostClassifier for the league
        
        Raises:
            ValueError if no model available
        """
        # Check cache
        if sport_key in self.models:
            return self.models[sport_key]
        
        # Get model file name
        model_file = SPORT_KEY_TO_MODEL.get(sport_key)
        
        if model_file:
            model = self._load_model(model_file)
            if model:
                self.models[sport_key] = model
                return model
        
        # Fallback to global V2 model
        if self.fallback_model is None:
            fallback_path = self.models_dir / "catboost_v2.cbm"
            if fallback_path.exists():
                self.fallback_model = self._load_model("catboost_v2.cbm")
        
        if self.fallback_model:
            logger.warning(f"Using global fallback model for {sport_key}")
            return self.fallback_model
        
        raise ValueError(f"No model available for sport_key: {sport_key}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature DataFrame using contract."""
        X = df[self.contract.features].copy()
        
        # Fill NaN
        for col in self.contract.numeric:
            if col in X.columns:
                X[col] = X[col].fillna(0.0)
        
        for col in self.contract.categorical:
            if col in X.columns:
                X[col] = X[col].fillna('Unknown').astype(str)
        
        return X
    
    def predict(
        self,
        df: pd.DataFrame,
        sport_key: str = 'soccer_epl'
    ) -> np.ndarray:
        """
        Predict probabilities for matches.
        
        Args:
            df: DataFrame with feature columns
            sport_key: API sport key
        
        Returns:
            Probability array (n_samples, 3) for H/D/A
        """
        model = self.get_model(sport_key)
        X = self.prepare_features(df)
        
        proba = model.predict_proba(X)
        return proba
    
    def predict_with_model_info(
        self,
        df: pd.DataFrame,
        sport_key: str = 'soccer_epl'
    ) -> Dict[str, Any]:
        """
        Predict with additional model metadata.
        
        Returns dict with:
            - proba: probability array
            - model_used: which model file was used
            - decay_half_life: decay used in training
        """
        model_file = SPORT_KEY_TO_MODEL.get(sport_key, 'catboost_v2.cbm')
        proba = self.predict(df, sport_key)
        
        # Get decay info from metadata
        decay_info = None
        if self.metadata and 'leagues' in self.metadata:
            for league_key, info in self.metadata['leagues'].items():
                if model_file.replace('catboost_', '').replace('.cbm', '') in league_key:
                    decay_info = info.get('decay_half_life')
                    break
        
        return {
            'proba': proba,
            'model_used': model_file,
            'sport_key': sport_key,
            'decay_half_life': decay_info,
        }
    
    def list_available_models(self) -> Dict[str, str]:
        """List all available model files and their sport key mappings."""
        available = {}
        
        for sport_key, model_file in SPORT_KEY_TO_MODEL.items():
            path = self.models_dir / model_file
            status = "✅" if path.exists() else "❌"
            available[sport_key] = f"{status} {model_file}"
        
        return available


def test_loader():
    """Test the league model loader."""
    print("\n" + "=" * 60)
    print("LEAGUE MODEL LOADER TEST")
    print("=" * 60)
    
    loader = LeagueModelLoader()
    
    print("\n📋 Available Models:")
    for sport_key, status in loader.list_available_models().items():
        print(f"  {sport_key:<35} {status}")
    
    # Test loading EPL model
    print("\n🔧 Testing EPL model load...")
    try:
        model = loader.get_model('soccer_epl')
        print(f"  ✅ Loaded EPL model: {type(model).__name__}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    print("\n✅ Loader test complete")


if __name__ == "__main__":
    test_loader()
