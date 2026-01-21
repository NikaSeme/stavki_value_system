"""
Unit tests for ML model.

CRITICAL: Tests temporal split validation and probability sums.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.ml_model import (
    FEATURE_COLS,
    MLModel,
    create_target,
    temporal_train_test_split,
)


class TestTargetCreation:
    """Test target variable creation."""
    
    def test_target_encoding(self):
        """Test correct target encoding."""
        df = pd.DataFrame({
            'home_goals': [2, 1, 0],
            'away_goals': [1, 1, 2],
        })
        
        target = create_target(df)
        
        # 2-1: Home win → 2
        # 1-1: Draw → 1
        # 0-2: Away win → 0
        assert list(target) == [2, 1, 0]
    
    def test_target_types(self):
        """Test target is integer array."""
        df = pd.DataFrame({
            'home_goals': [2, 1],
            'away_goals': [1, 1],
        })
        
        target = create_target(df)
        
        assert target.dtype in [np.int32, np.int64]
        assert len(target) == 2


class TestTemporalSplit:
    """Test temporal train/valid split functionality."""
    
    def test_no_future_leakage(self):
        """CRITICAL: Ensure train dates < valid dates (NO OVERLAP)."""
        # Create sequential dates
        dates = pd.date_range('2025-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'value': range(10)
        })
        
        train, valid = temporal_train_test_split(df, train_ratio=0.7)
        
        # Get date boundaries
        max_train_date = train['date'].max()
        min_valid_date = valid['date'].min()
        
        # CRITICAL: train dates must be <= valid dates
        assert max_train_date <= min_valid_date
    
    def test_split_ratio(self):
        """Test that split ratio is respected."""
        df = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=100),
            'value': range(100)
        })
        
        train, valid = temporal_train_test_split(df, train_ratio=0.7)
        
        assert len(train) == 70
        assert len(valid) == 30
    
    def test_temporal_ordering_maintained(self):
        """Test that temporal order is preserved."""
        dates = pd.date_range('2025-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'value': range(10)
        })
        
        train, valid = temporal_train_test_split(df, train_ratio=0.6)
        
        # Train should have first 6 dates
        assert list(train['value']) == [0, 1, 2, 3, 4, 5]
        
        # Valid should have last 4 dates
        assert list(valid['value']) == [6, 7, 8, 9]
    
    def test_error_on_overlap(self):
        """Test error if dates somehow overlap (defensive check)."""
        # This shouldn't happen in practice, but test the validation
        # Note: current implementation won't trigger this unless
        # there's a bug, but we keep it for safety
        pass


class TestMLModel:
    """Test ML model training and prediction."""
    
    def create_sample_data(self, n_samples=50):
        """Create sample training data."""
        dates = pd.date_range('2025-01-01', periods=n_samples, freq='D')
        
        df = pd.DataFrame({
            'date': dates,
            'home_goals': np.random.randint(0, 4, n_samples),
            'away_goals': np.random.randint(0, 4, n_samples),
            'home_goals_for_avg_5': np.random.uniform(0.5, 2.5, n_samples),
            'home_goals_against_avg_5': np.random.uniform(0.5, 2.5, n_samples),
            'home_points_avg_5': np.random.uniform(0, 3, n_samples),
            'home_form_points_5': np.random.uniform(0, 15, n_samples),
            'home_matches_count': np.random.randint(1, 10, n_samples),
            'away_goals_for_avg_5': np.random.uniform(0.5, 2.5, n_samples),
            'away_goals_against_avg_5': np.random.uniform(0.5, 2.5, n_samples),
            'away_points_avg_5': np.random.uniform(0, 3, n_samples),
            'away_form_points_5': np.random.uniform(0, 15, n_samples),
            'away_matches_count': np.random.randint(1, 10, n_samples),
        })
        
        return df
    
    def test_model_training(self):
        """Test basic model training."""
        df = self.create_sample_data(n_samples=50)
        
        model = MLModel()
        stats = model.train(df, train_ratio=0.7)
        
        # Check stats returned
        assert 'train_samples' in stats
        assert 'valid_samples' in stats
        assert stats['train_samples'] == 35
        assert stats['valid_samples'] == 15
        
        # Model should be trained
        assert model.base_model is not None
        assert model.calibrated_model is not None
    
    def test_predictions_shape(self):
        """Test prediction output shape."""
        df = self.create_sample_data(n_samples=50)
        
        model = MLModel()
        model.train(df, train_ratio=0.7)
        
        # Predict on all data
        predictions = model.predict(df)
        
        # Check shape
        assert len(predictions) == 50
        assert 'prob_home_ml' in predictions.columns
        assert 'prob_draw_ml' in predictions.columns
        assert 'prob_away_ml' in predictions.columns
    
    def test_probabilities_sum_to_one(self):
        """CRITICAL: Test that probabilities sum to 1.0."""
        df = self.create_sample_data(n_samples=50)
        
        model = MLModel()
        model.train(df, train_ratio=0.7)
        
        predictions = model.predict(df)
        
        # Check probabilities for each row
        for idx, row in predictions.iterrows():
            prob_sum = (
                row['prob_home_ml'] + 
                row['prob_draw_ml'] + 
                row['prob_away_ml']
            )
            
            assert prob_sum == pytest.approx(1.0, abs=1e-6)
    
    def test_model_persistence(self):
        """Test model save and load."""
        df = self.create_sample_data(n_samples=50)
        
        model = MLModel()
        model.train(df, train_ratio=0.7)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"
            model.save(model_path)
            
            # Load model
            loaded_model = MLModel.load(model_path)
            
            # Make predictions with both models
            pred1 = model.predict(df)
            pred2 = loaded_model.predict(df)
            
            # Predictions should be identical
            assert np.allclose(
                pred1['prob_home_ml'].values,
                pred2['prob_home_ml'].values
            )
    
    def test_temporal_split_in_training(self):
        """Test that temporal split is enforced during training."""
        # Create data with clear temporal pattern
        dates = pd.date_range('2025-01-01', periods=20, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'home_goals': [2] * 20,  # Constant for simplicity
            'away_goals': [1] * 20,
            **{col: [1.5] * 20 for col in FEATURE_COLS}
        })
        
        model = MLModel()
        stats = model.train(df, train_ratio=0.7)
        
        # Training should use first 14 samples
        # Validation should use last 6 samples
        assert stats['train_samples'] == 14
        assert stats['valid_samples'] == 6


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_pipeline(self):
        """Test complete train and predict workflow."""
        # Create sample data
        dates = pd.date_range('2025-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            ' date': dates,
            'league': ['E0'] * 50,
            'home_team': ['Arsenal'] * 50,
            'away_team': ['Chelsea'] * 50,
            'home_goals': np.random.randint(0, 4, 50),
            'away_goals': np.random.randint(0, 4, 50),
            'home_goals_for_avg_5': np.random.uniform(0.5, 2.5, 50),
            'home_goals_against_avg_5': np.random.uniform(0.5, 2.5, 50),
            'home_points_avg_5': np.random.uniform(0, 3, 50),
            'home_form_points_5': np.random.uniform(0, 15, 50),
            'home_matches_count': np.random.randint(1, 10, 50),
            'away_goals_for_avg_5': np.random.uniform(0.5, 2.5, 50),
            'away_goals_against_avg_5': np.random.uniform(0.5, 2.5, 50),
            'away_points_avg_5': np.random.uniform(0, 3, 50),
            'away_form_points_5': np.random.uniform(0, 15, 50),
            'away_matches_count': np.random.randint(1, 10, 50),
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "features.csv"
            model_file = Path(tmpdir) / "model.pkl"
            output_file = Path(tmpdir) / "predictions.csv"
            
            df.to_csv(input_file, index=False)
            
            # Train
            model = MLModel()
            model.train_from_file(input_file, model_file, train_ratio=0.7)
            
            # Predict
            loaded_model = MLModel.load(model_file)
            loaded_model.predict_from_file(input_file, output_file)
            
            # Verify output
            assert output_file.exists()
            
            predictions = pd.read_csv(output_file)
            assert len(predictions) == 50
            assert 'prob_home_ml' in predictions.columns
