import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import joblib
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.models.calibration import SafeCalibrator
from src.logging_setup import get_logger

logger = get_logger(__name__)


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
        self.scaler = None  # Can be StandardScaler or ColumnTransformer (preprocessor)
        self.preprocessor = None  # Alias for scaler (clearer naming for ColumnTransformer)
        self.metadata: Optional[Dict[str, Any]] = None
        self._feature_names: Optional[List[str]] = None
    
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
                raise RuntimeError(
                    f"CRITICAL: Model file missing: {model_file}. "
                    "Run scripts/train_model.py first."
                )
            
            logger.info(f"Loading model from {model_file}...")
            self.model = joblib.load(model_file)
            logger.info(f"  ✓ Loaded model ({self.model.__class__.__name__})")
            
            # Load calibrator
            calib_file = self.models_dir / 'calibrator_v1_latest.pkl'
            if not calib_file.exists():
                raise RuntimeError(f"CRITICAL: Calibrator missing: {calib_file}")
            
            logger.info(f"Loading calibrator from {calib_file}...")
            self.calibrator = joblib.load(calib_file)
            logger.info(f"  ✓ Loaded calibrator")
            
            # Load scaler
            scaler_file = self.models_dir / 'scaler_v1_latest.pkl'
            if not scaler_file.exists():
                raise RuntimeError(f"CRITICAL: Scaler missing: {scaler_file}")
            
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
            raise RuntimeError(f"CRITICAL: Failed to load model artifacts: {e}")
    
    def validate(self) -> bool:
        """
        Validate that all required artifacts are loaded.
        
        Returns:
            True if valid
        
        Raises:
            RuntimeError: If validation fails
        """
        if self.model is None:
            raise RuntimeError("Validation failed: Model not loaded")
        
        if self.calibrator is None:
            raise RuntimeError("Validation failed: Calibrator not loaded")
        
        if self.scaler is None:
            raise RuntimeError("Validation failed: Scaler not loaded")
        
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
        n_features = X.shape[1]
        if n_features != self.scaler.n_features_in_:
            raise ValueError(
                f"Feature count mismatch: got {n_features}, "
                f"expected {self.scaler.n_features_in_}"
            )
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict with calibrated model
        probs = self.calibrator.predict_proba(X_scaled)
        
        return probs
    
    def get_feature_names(self) -> Optional[List[str]]:
        """
        Get feature names from metadata.
        
        Returns:
            List of feature names if available, None otherwise
        """
        if self._feature_names:
            return self._feature_names
        if self.metadata:
            self._feature_names = self.metadata.get('features', None)
            return self._feature_names
        return None
    
    def get_preprocessor(self):
        """
        Get the preprocessor (scaler/ColumnTransformer).
        
        Returns:
            The fitted preprocessor for feature transformation
        """
        if self.preprocessor is not None:
            return self.preprocessor
        if self.scaler is not None:
            return self.scaler
        raise RuntimeError("No preprocessor loaded. Call load_latest() first.")
    
    def validate_features(self, feature_names: List[str]) -> bool:
        """
        Validate that provided features match training features.
        
        Args:
            feature_names: List of feature names to validate
            
        Returns:
            True if features match, False if mismatch detected
            
        Raises:
            RuntimeError: If metadata is missing
        """
        expected = self.get_feature_names()
        if expected is None:
            logger.warning("No feature names in metadata - skipping validation")
            return True
        
        # Check for exact match
        if feature_names == expected:
            logger.debug("Feature validation passed: exact match")
            return True
        
        # Find differences
        missing = set(expected) - set(feature_names)
        extra = set(feature_names) - set(expected)
        
        if missing:
            logger.error(f"Missing features (required for inference): {missing}")
        if extra:
            logger.warning(f"Extra features (will be ignored): {extra}")
        
        if missing:
            raise ValueError(
                f"Feature mismatch: {len(missing)} missing features. "
                f"First 5: {list(missing)[:5]}"
            )
        
        return True


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
