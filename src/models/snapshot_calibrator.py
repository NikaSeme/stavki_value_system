"""
Multi-class Probability Calibration

Shared module for calibration class to ensure pickle compatibility.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
import logging

logger = logging.getLogger(__name__)


class MultiClassCalibrator:
    """
    Isotonic calibration for multi-class probabilities.
    Fits one isotonic regressor per class, then renormalizes.
    """
    
    def __init__(self):
        self.calibrators = {}
        self.is_fitted = False
    
    def fit(self, y_true: np.ndarray, y_proba: np.ndarray):
        """
        Fit calibrators on validation data.
        
        Args:
            y_true: True labels (0, 1, 2)
            y_proba: Raw probabilities from model (n_samples, 3)
        """
        n_classes = y_proba.shape[1]
        
        for c in range(n_classes):
            binary = (y_true == c).astype(float)
            
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(y_proba[:, c], binary)
            self.calibrators[c] = iso
            
            logger.info(f"Fitted calibrator for class {c}")
        
        self.is_fitted = True
    
    def predict_proba(self, y_proba: np.ndarray) -> np.ndarray:
        """
        Apply calibration and renormalize.
        
        Args:
            y_proba: Raw probabilities (n_samples, 3)
        
        Returns:
            Calibrated probabilities summing to 1.0
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        
        n_samples, n_classes = y_proba.shape
        calibrated = np.zeros_like(y_proba)
        
        for c in range(n_classes):
            calibrated[:, c] = self.calibrators[c].predict(y_proba[:, c])
        
        # Renormalize to sum to 1.0
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        calibrated = calibrated / row_sums
        
        return calibrated
