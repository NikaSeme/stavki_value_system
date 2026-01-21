"""
Probability calibration for match outcome predictions.

Implements isotonic per-class calibration with renormalization to ensure
probabilities sum exactly to 1.0.
"""

from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ..logging_setup import get_logger

logger = get_logger(__name__)


def renormalize_probabilities(probs: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """
    Renormalize probabilities to sum exactly to 1.0.
    
    Args:
        probs: Probability matrix (n_samples, n_classes)
        epsilon: Small value to avoid division by zero
        
    Returns:
        Renormalized probabilities that sum to 1.0 per row
    """
    # Calculate row sums
    row_sums = probs.sum(axis=1, keepdims=True)
    
    # Handle edge case: all zeros - distribute uniformly
    n_classes = probs.shape[1]
    mask = row_sums.ravel() < epsilon
    
    # Normalize non-zero rows
    normalized = probs / np.where(row_sums < epsilon, 1.0, row_sums)
    
    # For all-zero rows, assign uniform probabilities
    normalized[mask] = 1.0 / n_classes
    
    return normalized


class IsotonicCalibrator:
    """
    Isotonic per-class probability calibrator.
    
    Fits separate isotonic regression for each class, then renormalizes
    to ensure probabilities sum to 1.0.
    """
    
    def __init__(self, n_classes: int = 3):
        """
        Initialize calibrator.
        
        Args:
            n_classes: Number of classes (default: 3 for Home/Draw/Away)
        """
        self.n_classes = n_classes
        self.calibrators = [
            IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            for _ in range(n_classes)
        ]
        self.is_fitted = False
        
        logger.info(f"Initialized IsotonicCalibrator with {n_classes} classes")
    
    def fit(self, y_true: np.ndarray, y_probs: np.ndarray) -> 'IsotonicCalibrator':
        """
        Fit isotonic calibrators on validation data.
        
        Args:
            y_true: True class labels (n_samples,)
            y_probs: Predicted probabilities (n_samples, n_classes)
            
        Returns:
            Self (fitted)
        """
        logger.info(f"Fitting isotonic calibration on {len(y_true)} samples")
        
        if y_probs.shape[1] != self.n_classes:
            raise ValueError(
                f"y_probs has {y_probs.shape[1]} classes, "
                f"expected {self.n_classes}"
            )
        
        # Fit isotonic regressor for each class
        for class_idx in range(self.n_classes):
            # Create binary labels: 1 if true class, 0 otherwise
            binary_labels = (y_true == class_idx).astype(float)
            
            # Fit isotonic regression
            self.calibrators[class_idx].fit(
                y_probs[:, class_idx],
                binary_labels
            )
        
        self.is_fitted = True
        logger.info("Isotonic calibration fitted")
        
        return self
    
    def predict_proba(self, y_probs: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration and renormalize.
        
        Args:
            y_probs: Uncalibrated probabilities (n_samples, n_classes)
            
        Returns:
            Calibrated and renormalized probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        # Apply isotonic calibration per class
        calibrated = np.column_stack([
            self.calibrators[i].predict(y_probs[:, i])
            for i in range(self.n_classes)
        ])
        
        # Renormalize to ensure sum = 1.0
        calibrated = renormalize_probabilities(calibrated)
        
        # Verify sum
        sums = calibrated.sum(axis=1)
        max_deviation = np.abs(sums - 1.0).max()
        
        if max_deviation > 1e-9:
            logger.warning(
                f"Max probability sum deviation: {max_deviation:.2e} "
                f"(expected < 1e-9)"
            )
        
        return calibrated


class CalibratedModel:
    """
    Wrapper for any model with isotonic calibration.
    
    Example usage:
        model = CalibratedModel(base_model)
        model.fit_calibration(X_valid, y_valid)
        calibrated_probs = model.predict_proba(X_test)
    """
    
    def __init__(self, base_model, n_classes: int = 3):
        """
        Initialize calibrated model.
        
        Args:
            base_model: Model with predict_proba() method
            n_classes: Number of classes
        """
        self.base_model = base_model
        self.calibrator = IsotonicCalibrator(n_classes=n_classes)
        
    def fit_calibration(self, X_valid, y_valid) -> 'CalibratedModel':
        """
        Fit calibration on validation set.
        
        Args:
            X_valid: Validation features
            y_valid: Validation labels
            
        Returns:
            Self (with fitted calibration)
        """
        # Get uncalibrated predictions
        uncalibrated_probs = self.base_model.predict_proba(X_valid)
        
        # Fit calibrator
        self.calibrator.fit(y_valid, uncalibrated_probs)
        
        return self
    
    def predict_proba(self, X):
        """
        Predict with calibrated probabilities.
        
        Args:
            X: Features
            
        Returns:
            Calibrated probabilities
        """
        # Get base predictions
        base_probs = self.base_model.predict_proba(X)
        
        # Apply calibration
        calibrated_probs = self.calibrator.predict_proba(base_probs)
        
        return calibrated_probs
