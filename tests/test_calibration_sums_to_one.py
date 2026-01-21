"""
Unit tests for probability calibration.

CRITICAL: Tests that probabilities sum to exactly 1.0 (tolerance: 1e-9).
"""

import numpy as np
import pytest

from src.models.calibration import (
    IsotonicCalibrator,
    renormalize_probabilities,
)


class TestRenormalization:
    """Test probability renormalization."""
    
    def test_renormalization_sums_to_one(self):
        """CRITICAL: Renormalized probabilities must sum to 1.0."""
        # Create some unnormalized probabilities
        probs = np.array([
            [0.4, 0.3, 0.2],  # Sum = 0.9
            [0.5, 0.5, 0.5],  # Sum = 1.5
            [0.1, 0.1, 0.1],  # Sum = 0.3
        ])
        
        normalized = renormalize_probabilities(probs)
        
        # Check each row sums to 1.0 with high precision
        for row in normalized:
            prob_sum = row.sum()
            assert abs(prob_sum - 1.0) < 1e-9, f"Sum {prob_sum} deviates from 1.0"
    
    def test_already_normalized(self):
        """Test that already normalized probs stay normalized."""
        probs = np.array([
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
        ])
        
        normalized = renormalize_probabilities(probs)
        
        # Should be very close to original
        np.testing.assert_allclose(normalized, probs, atol=1e-10)
        
        # Still sum to 1.0
        for row in normalized:
            assert abs(row.sum() - 1.0) < 1e-9
    
    def test_edge_case_all_zeros(self):
        """Test edge case: all zeros in a row."""
        probs = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.3, 0.2],
        ])
        
        normalized = renormalize_probabilities(probs)
        
        # First row should become uniform
        assert abs(normalized[0].sum() - 1.0) < 1e-9
        
        # Second row should be unchanged
        np.testing.assert_allclose(normalized[1], probs[1], atol=1e-10)
    
    def test_extreme_values(self):
        """Test with very small and very large values."""
        probs = np.array([
            [1e-10, 1e-10, 0.999],
            [0.999, 1e-10, 1e-10],
        ])
        
        normalized = renormalize_probabilities(probs)
        
        for row in normalized:
            assert abs(row.sum() - 1.0) < 1e-9


class TestIsotonicCalibrator:
    """Test isotonic calibration."""
    
    def test_fit_and_predict(self):
        """Test basic fit and predict workflow."""
        # Create synthetic data
        np.random.seed(42)
        n_samples = 100
        
        # Simulated uncalibrated probabilities
        probs = np.random.dirichlet([2, 1, 1], size=n_samples)
        
        # True labels (somewhat aligned with probs)
        y_true = np.array([
            np.random.choice(3, p=p) for p in probs
        ])
        
        # Fit calibrator
        calibrator = IsotonicCalibrator(n_classes=3)
        calibrator.fit(y_true, probs)
        
        # Predict
        calibrated = calibrator.predict_proba(probs)
        
        # Check shape
        assert calibrated.shape == probs.shape
        
        # Check probabilities sum to 1.0
        for row in calibrated:
            assert abs(row.sum() - 1.0) < 1e-9
    
    def test_probabilities_in_valid_range(self):
        """Test that calibrated probabilities are in [0, 1]."""
        np.random.seed(42)
        n_samples = 50
        
        probs = np.random.dirichlet([1, 1, 1], size=n_samples)
        y_true = np.random.randint(0, 3, size=n_samples)
        
        calibrator = IsotonicCalibrator(n_classes=3)
        calibrator.fit(y_true, probs)
        
        calibrated = calibrator.predict_proba(probs)
        
        # All values should be in [0, 1]
        assert (calibrated >= 0).all()
        assert (calibrated <= 1).all()
    
    def test_calibration_preserves_sum(self):
        """CRITICAL: Calibration must preserve probability sum = 1.0."""
        np.random.seed(42)
        
        # Create well-formed probabilities
        probs = np.array([
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
            [0.3, 0.3, 0.4],
            [0.6, 0.2, 0.2],
        ])
        
        y_true = np.array([0, 1, 2, 0])
        
        calibrator = IsotonicCalibrator(n_classes=3)
        calibrator.fit(y_true, probs)
        
        calibrated = calibrator.predict_proba(probs)
        
        # Every row must sum to 1.0 with tolerance 1e-9
        for i, row in enumerate(calibrated):
            prob_sum = row.sum()
            assert abs(prob_sum - 1.0) < 1e-9, \
                f"Row {i}: sum={prob_sum}, deviation={abs(prob_sum - 1.0)}"
    
    def test_error_if_not_fitted(self):
        """Test error when predicting without fitting."""
        calibrator = IsotonicCalibrator(n_classes=3)
        
        probs = np.array([[0.5, 0.3, 0.2]])
        
        with pytest.raises(ValueError, match="not fitted"):
            calibrator.predict_proba(probs)
    
    def test_different_class_counts(self):
        """Test with different number of classes."""
        for n_classes in [2, 3, 4]:
            np.random.seed(42)
            
            # Generate random probabilities
            probs = np.random.dirichlet([1]*n_classes, size=20)
            y_true = np.random.randint(0, n_classes, size=20)
            
            calibrator = IsotonicCalibrator(n_classes=n_classes)
            calibrator.fit(y_true, probs)
            
            calibrated = calibrator.predict_proba(probs)
            
            # Check sums
            for row in calibrated:
                assert abs(row.sum() - 1.0) < 1e-9


class TestEnsembleProbabilities:
    """Test ensemble model probability sums."""
    
    def test_ensemble_output_format(self):
        """Test that ensemble produces valid probability distributions."""
        from src.models.ensemble import EnsembleModel
        import pandas as pd
        
        np.random.seed(42)
        n_samples = 50
        
        # Create synthetic base model predictions
        poisson_probs = pd.DataFrame({
            'prob_home': np.random.uniform(0.2, 0.6, n_samples),
            'prob_draw': np.random.uniform(0.2, 0.4, n_samples),
            'prob_away': np.random.uniform(0.1, 0.5, n_samples),
        })
        # Normalize
        row_sums = poisson_probs.sum(axis=1)
        poisson_probs = poisson_probs.div(row_sums, axis=0)
        
        ml_probs = pd.DataFrame({
            'prob_home_ml': np.random.uniform(0.2, 0.6, n_samples),
            'prob_draw_ml': np.random.uniform(0.2, 0.4, n_samples),
            'prob_away_ml': np.random.uniform(0.1, 0.5, n_samples),
        })
        # Normalize
        row_sums = ml_probs.sum(axis=1)
        ml_probs = ml_probs.div(row_sums, axis=0)
        
        # Generate true labels
        y_true = np.random.randint(0, 3, size=n_samples)
        
        # Train ensemble
        ensemble = EnsembleModel(calibrate=True)
        ensemble.train(poisson_probs, ml_probs, y_true, calibration_split=0.3)
        
        # Predict
        ensemble_probs = ensemble.predict_proba(poisson_probs, ml_probs)
        
        # CRITICAL: Check probabilities sum to 1.0
        for i, row in enumerate(ensemble_probs):
            prob_sum = row.sum()
            assert abs(prob_sum - 1.0) < 1e-9, \
                f"Row {i}: sum={prob_sum}, deviation={abs(prob_sum - 1.0)}"
    
    def test_ensemble_without_calibration(self):
        """Test ensemble without calibration still sums to 1.0."""
        from src.models.ensemble import EnsembleModel
        import pandas as pd
        
        np.random.seed(42)
        n_samples = 30
        
        # Create base predictions
        poisson_probs = pd.DataFrame({
            'prob_home': np.random.dirichlet([2, 1, 1], n_samples)[:, 0],
            'prob_draw': np.random.dirichlet([2, 1, 1], n_samples)[:, 1],
            'prob_away': np.random.dirichlet([2, 1, 1], n_samples)[:, 2],
        })
        
        ml_probs = pd.DataFrame({
            'prob_home_ml': np.random.dirichlet([1, 1, 2], n_samples)[:, 0],
            'prob_draw_ml': np.random.dirichlet([1, 1, 2], n_samples)[:, 1],
            'prob_away_ml': np.random.dirichlet([1, 1, 2], n_samples)[:, 2],
        })
        
        y_true = np.random.randint(0, 3, size=n_samples)
        
        # Train without calibration
        ensemble = EnsembleModel(calibrate=False)
        ensemble.train(poisson_probs, ml_probs, y_true)
        
        # Predict
        ensemble_probs = ensemble.predict_proba(poisson_probs, ml_probs)
        
        # Still should sum to 1.0 (LogisticRegression outputs sum to 1.0)
        for row in ensemble_probs:
            assert abs(row.sum() - 1.0) < 1e-9
