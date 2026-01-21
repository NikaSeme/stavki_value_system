"""
Ensemble model for combining Poisson and ML predictions.

Uses stacking with multinomial logistic regression as meta-model.
"""

import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .calibration import IsotonicCalibrator, renormalize_probabilities
from ..logging_setup import get_logger

logger = get_logger(__name__)


def stack_predictions(
    poisson_probs: pd.DataFrame,
    ml_probs: pd.DataFrame
) -> np.ndarray:
    """
    Stack base model predictions as features for meta-model.
    
    Args:
        poisson_probs: DataFrame with prob_home, prob_draw, prob_away
        ml_probs: DataFrame with prob_home_ml, prob_draw_ml, prob_away_ml
        
    Returns:
        Stacked feature matrix (n_samples, 6)
    """
    features = np.column_stack([
        poisson_probs['prob_home'].values,
        poisson_probs['prob_draw'].values,
        poisson_probs['prob_away'].values,
        ml_probs['prob_home_ml'].values,
        ml_probs['prob_draw_ml'].values,
        ml_probs['prob_away_ml'].values,
    ])
    
    return features


class EnsembleModel:
    """
    Ensemble model using stacking with calibration.
    
    Combines Poisson (Model A) and ML (Model B) predictions using
    multinomial logistic regression, then applies isotonic calibration.
    """
    
    def __init__(
        self,
        max_iter: int = 1000,
        calibrate: bool = True
    ):
        """
        Initialize ensemble model.
        
        Args:
            max_iter: Maximum iterations for logistic regression
            calibrate: Whether to apply isotonic calibration
        """
        self.max_iter = max_iter
        self.calibrate = calibrate
        
        self.meta_model: Optional[LogisticRegression] = None
        self.calibrator: Optional[IsotonicCalibrator] = None
        
        logger.info(
            f"Initialized EnsembleModel: "
            f"max_iter={max_iter}, calibrate={calibrate}"
        )
    
    def train(
        self,
        poisson_probs: pd.DataFrame,
        ml_probs: pd.DataFrame,
        y_true: np.ndarray,
        calibration_split: float = 0.3
    ) -> Dict[str, any]:
        """
        Train meta-model and optionally fit calibration.
        
        Args:
            poisson_probs: Poisson predictions
            ml_probs: ML predictions
            y_true: True labels (0=Away, 1=Draw, 2=Home)
            calibration_split: Fraction of data for calibration
            
        Returns:
            Training statistics
        """
        logger.info(f"Training ensemble on {len(y_true)} samples")
        
        # Stack features
        X = stack_predictions(poisson_probs, ml_probs)
        
        # Split for meta-model training and calibration
        if self.calibrate and calibration_split > 0:
            split_idx = int(len(X) * (1 - calibration_split))
            
            X_train = X[:split_idx]
            y_train = y_true[:split_idx]
            
            X_cal = X[split_idx:]
            y_cal = y_true[split_idx:]
            
            logger.info(
                f"Meta-model train: {len(X_train)}, "
                f"calibration: {len(X_cal)}"
            )
        else:
            X_train = X
            y_train = y_true
            X_cal = None
            y_cal = None
        
        # Train meta-model
        logger.info("Training multinomial logistic regression meta-model...")
        self.meta_model = LogisticRegression(
            solver='lbfgs',
            max_iter=self.max_iter,
            random_state=42
        )
        
        self.meta_model.fit(X_train, y_train)
        
        # Train accuracy
        train_pred = self.meta_model.predict(X_train)
        train_acc = (train_pred == y_train).mean()
        
        # Calibration
        if self.calibrate and X_cal is not None:
            logger.info("Fitting isotonic calibration...")
            
            # Get uncalibrated probabilities on calibration set
            uncal_probs = self.meta_model.predict_proba(X_cal)
            
            # Fit calibrator
            self.calibrator = IsotonicCalibrator(n_classes=3)
            self.calibrator.fit(y_cal, uncal_probs)
            
            # Calibration accuracy
            cal_probs = self.calibrator.predict_proba(uncal_probs)
            cal_pred = cal_probs.argmax(axis=1)
            cal_acc = (cal_pred == y_cal).mean()
            
            logger.info(f"Calibration accuracy: {cal_acc:.3f}")
        
        stats = {
            'train_samples': len(X_train),
            'train_accuracy': float(train_acc),
            'calibrated': self.calibrate,
        }
        
        if self.calibrate and X_cal is not None:
            stats['calibration_samples'] = len(X_cal)
            stats['calibration_accuracy'] = float(cal_acc)
        
        logger.info("Ensemble training complete")
        
        return stats
    
    def predict_proba(
        self,
        poisson_probs: pd.DataFrame,
        ml_probs: pd.DataFrame
    ) -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Args:
            poisson_probs: Poisson predictions
            ml_probs: ML predictions
            
        Returns:
            Ensemble probabilities (n_samples, 3)
        """
        if self.meta_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Stack features
        X = stack_predictions(poisson_probs, ml_probs)
        
        # Get meta-model predictions
        probs = self.meta_model.predict_proba(X)
        
        # Apply calibration if fitted
        if self.calibrate and self.calibrator is not None:
            probs = self.calibrator.predict_proba(probs)
        
        # Ensure probabilities sum to 1.0
        probs = renormalize_probabilities(probs)
        
        # Verify
        sums = probs.sum(axis=1)
        max_deviation = np.abs(sums - 1.0).max()
        
        if max_deviation > 1e-9:
            logger.warning(
                f"Probability sum deviation: {max_deviation:.2e}"
            )
        
        return probs
    
    def save(self, filepath: Path) -> None:
        """
        Save ensemble model.
        
        Args:
            filepath: Path to save model
        """
        if self.meta_model is None:
            raise ValueError("No model to save. Train first.")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'meta_model': self.meta_model,
            'calibrator': self.calibrator,
            'max_iter': self.max_iter,
            'calibrate': self.calibrate,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Ensemble model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'EnsembleModel':
        """
        Load ensemble model.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded EnsembleModel
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            max_iter=model_data['max_iter'],
            calibrate=model_data['calibrate']
        )
        
        model.meta_model = model_data['meta_model']
        model.calibrator = model_data['calibrator']
        
        logger.info(f"Ensemble model loaded from {filepath}")
        
        return model
