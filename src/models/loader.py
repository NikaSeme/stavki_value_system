"""
Model loader for trained ML models.

Loads CatBoost model, calibrator, and scaler with strict validation.
"""

import joblib
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Load and manage trained ML models.
    
    Provides strict validation and fail-hard behavior if models are missing.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model loader.
        
        Args:
            models_dir: Directory containing model artifacts
        """
        self.models_dir = Path(models_dir)
        self.model = None
        self.calibrator = None
        self.scaler = None
        self.metadata: Optional[Dict[str, Any]] = None
    
    def load_latest(self) -> bool:
        """
        Load latest model artifacts.
        
        Looks for *_latest.pkl symlinks in models directory.
        
        Returns:
            True if all artifacts loaded successfully, False otherwise
        """
        try:
            # Load model
            model_file = self.models_dir / 'catboost_v1_latest.pkl'
            if not model_file.exists():
                logger.error(f"Model file not found: {model_file}")
                logger.error(f"  Expected location: {model_file.absolute()}")
                logger.error(f"  Run: python scripts/train_model.py")
                return False
            
            logger.info(f"Loading model from {model_file}...")
            self.model = joblib.load(model_file)
            logger.info(f"  ✓ Loaded model ({self.model.__class__.__name__})")
            
            # Load calibrator
            calib_file = self.models_dir / 'calibrator_v1_latest.pkl'
            if not calib_file.exists():
                logger.error(f"Calibrator not found: {calib_file}")
                return False
            
            logger.info(f"Loading calibrator from {calib_file}...")
            self.calibrator = joblib.load(calib_file)
            logger.info(f"  ✓ Loaded calibrator")
            
            # Load scaler
            scaler_file = self.models_dir / 'scaler_v1_latest.pkl'
            if not scaler_file.exists():
                logger.error(f"Scaler not found: {scaler_file}")
                return False
            
            logger.info(f"Loading scaler from {scaler_file}...")
            self.scaler = joblib.load(scaler_file)
            logger.info(f"  ✓ Loaded scaler")
            
            # Load metadata (optional)
            meta_file = self.models_dir / 'metadata_v1_latest.json'
            if meta_file.exists():
                with open(meta_file) as f:
                    self.metadata = json.load(f)
                logger.info(f"  ✓ Loaded metadata")
                logger.info(f"    Model version: {self.metadata.get('version', 'unknown')}")
                logger.info(f"    Features: {self.metadata.get('num_features', 'unknown')}")
                logger.info(f"    Test accuracy: {self.metadata['metrics']['test']['accuracy']:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate(self) -> bool:
        """
        Validate that all required artifacts are loaded.
        
        Returns:
            True if valid, False otherwise
        """
        if self.model is None:
            logger.error("Validation failed: Model not loaded")
            return False
        
        if self.calibrator is None:
            logger.error("Validation failed: Calibrator not loaded")
            return False
        
        if self.scaler is None:
            logger.error("Validation failed: Scaler not loaded")
            return False
        
        # Check that scaler has expected feature count
        if self.metadata:
            expected_features = self.metadata.get('num_features', 0)
            actual_features = self.scaler.n_features_in_
            
            if expected_features != actual_features:
                logger.warning(
                    f"Feature count mismatch: expected {expected_features}, "
                    f"scaler has {actual_features}"
                )
        
        logger.info("✓ Model validation passed")
        return True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict calibrated probabilities.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Calibrated probabilities (n_samples, 3) for Home/Draw/Away
            
        Raises:
            RuntimeError: If model not properly loaded
            ValueError: If feature count mismatch
        """
        if not self.validate():
            raise RuntimeError("Model not properly loaded or validated")
        
        # Check feature count
        if X.shape[1] != self.scaler.n_features_in_:
            raise ValueError(
                f"Feature count mismatch: got {X.shape[1]}, "
                f"expected {self.scaler.n_features_in_}"
            )
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict with calibrated model
        probs = self.calibrator.predict_proba(X_scaled)
        
        return probs
    
    def get_feature_names(self) -> Optional[list]:
        """Get feature names from metadata."""
        if self.metadata:
            return self.metadata.get('features', None)
        return None


def test_loader():
    """Test model loading."""
    loader = ModelLoader()
    
    print("Testing model loader...")
    success = loader.load_latest()
    
    if not success:
        print("❌ Failed to load model")
        return False
    
    if not loader.validate():
        print("❌ Model validation failed")
        return False
    
    print("\n✅ Model loader test passed!")
    print(f"   Model type: {loader.model.__class__.__name__}")
    print(f"   Features: {loader.scaler.n_features_in_}")
    
    if loader.metadata:
        print(f"   Trained: {loader.metadata.get('train_date', 'unknown')}")
        print(f"   Test accuracy: {loader.metadata['metrics']['test']['accuracy']:.2%}")
    
    # Test prediction with dummy data
    print("\nTesting prediction with dummy features...")
    X_dummy = np.random.randn(1, loader.scaler.n_features_in_)
    probs = loader.predict(X_dummy)
    
    print(f"   Probabilities: H={probs[0,0]:.3f}, D={probs[0,1]:.3f}, A={probs[0,2]:.3f}")
    print(f"   Sum: {probs[0].sum():.3f} (should be ~1.0)")
    
    return True


if __name__ == '__main__':
    test_loader()
