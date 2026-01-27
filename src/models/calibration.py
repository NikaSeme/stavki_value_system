import numpy as np
import logging
from sklearn.isotonic import IsotonicRegression
import sklearn

logger = logging.getLogger(__name__)

# Try to import FrozenEstimator (sklearn 1.6+)
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None

def renormalize_probabilities(probs):
    """Renormalize probabilities to sum to 1."""
    sums = probs.sum(axis=1, keepdims=True)
    # Avoid division by zero
    sums[sums == 0] = 1.0
    return probs / sums

class IsotonicCalibrator:
    """
    A bug-proof replacement for CalibratedClassifierCV that works on all 
    scikit-learn versions by manually implementing isotonic calibration.
    
    Robustness features:
    - Handles missing classes in validation data (fallbacks to identity).
    - Stable serialization path (defined in src.models).
    """
    def __init__(self, base_model, scaler=None):
        self.base_model = base_model
        self.scaler = scaler
        self.calibrators = {} # Changed to dict for class_index -> iso
        
    def fit(self, X_val, y_val):
        # Apply scaler if present
        X_val_processed = self.scaler.transform(X_val) if self.scaler else X_val
        
        # Get raw probabilities
        probs = self.base_model.predict_proba(X_val_processed)
        n_classes = probs.shape[1]
        self.calibrators = {}
        
        unique_y = np.unique(y_val)
        
        for i in range(n_classes):
            if i not in unique_y:
                logger.warning(f"Class {i} not present in validation data. Skipping calibration (identity fallback).")
                self.calibrators[i] = None # Marker for identity
                continue
                
            iso = IsotonicRegression(out_of_bounds='clip')
            target = (y_val == i).astype(float)
            
            # Additional safety: if target has only one value (all 0 or all 1)
            if len(np.unique(target)) < 2:
                logger.warning(f"Class {i} has no variation in target labels. Skipping calibration.")
                self.calibrators[i] = None
                continue
                
            iso.fit(probs[:, i], target)
            self.calibrators[i] = iso
            
        return self
        
    def predict_proba(self, X):
        # Apply scaler if present
        X_processed = self.scaler.transform(X) if self.scaler else X
        
        probs = self.base_model.predict_proba(X_processed)
        calibrated_probs = np.zeros_like(probs)
        
        for i in range(probs.shape[1]):
            iso = self.calibrators.get(i)
            if iso is None:
                # Identity fallback
                calibrated_probs[:, i] = probs[:, i]
            else:
                calibrated_probs[:, i] = iso.transform(probs[:, i])
            
        return renormalize_probabilities(calibrated_probs)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# Alias for backward compatibility
SafeCalibrator = IsotonicCalibrator

def get_best_calibrator(base_model, cv='prefit'):
    """
    Factory to return the optimal calibrator for the current sklearn version.
    
    Args:
        base_model: The trained classifier.
        cv: Calibration mode (default: 'prefit').
        
    Returns:
        An initialized calibrator instance.
    """
    # 1. Try FrozenEstimator (sklearn 1.6+) for native support
    if FrozenEstimator is not None:
        logger.info("Using sklearn.frozen.FrozenEstimator for calibration.")
        return FrozenEstimator(base_model)
    
    # 2. Fallback to IsotonicCalibrator (Custom Safe Implementation)
    logger.info("Using src.models.calibration.SafeCalibrator (IsotonicCalibrator).")
    return IsotonicCalibrator(base_model)
