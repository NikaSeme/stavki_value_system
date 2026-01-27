"""
Line Movement Sequence Model.

Predicts sharp line movements based on recent odds history.
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional

class LineSequenceModel(BaseEstimator, ClassifierMixin):
    """
    Model that predicts if odds will drop significantly based on recent history.
    """
    
    def __init__(self, n_lags=5, threshold_drop=0.05):
        """
        Args:
            n_lags: Number of past snapshots to use as features.
            threshold_drop: Target drop percentage to predict (e.g., 0.05 for 5%).
        """
        self.n_lags = n_lags
        self.threshold_drop = threshold_drop
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _extract_sequence_features(self, sequences: List[List[float]]) -> np.ndarray:
        """
        Convert raw odds sequences into feature matrix.
        
        Features:
        - Vel: Velocity (change between steps)
        - Acc: Acceleration (change of velocity)
        - Vol: Volatility (std dev)
        - Range: Max - Min
        """
        X = []
        
        for seq in sequences:
            # Ensure sequence length matches n_lags
            if len(seq) < self.n_lags:
                # Pad with first value
                seq = [seq[0]] * (self.n_lags - len(seq)) + list(seq)
            
            # Take last n_lags
            seq = np.array(seq[-self.n_lags:])
            
            # Features
            velocity = np.diff(seq)
            acceleration = np.diff(velocity)
            volatility = np.std(seq)
            total_range = np.max(seq) - np.min(seq)
            
            # Combine raw values + derived features
            features = np.concatenate([
                seq,          # Raw levels
                velocity,     # 1st deriv
                [volatility], # Vol
                [total_range] # Range
            ])
            
            X.append(features)
            
        return np.array(X)
        
    def fit(self, sequences: List[List[float]], labels: List[int]):
        """Train the model."""
        X = self._extract_sequence_features(sequences)
        y = np.array(labels)
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Train
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
        
    def predict_proba(self, sequences: List[List[float]]) -> np.ndarray:
        """Predict probabilities of sharp drop."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
            
        X = self._extract_sequence_features(sequences)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
        
    def predict(self, sequences: List[List[float]]) -> np.ndarray:
        """Predict binary class."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
            
        X = self._extract_sequence_features(sequences)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self, path: str):
        """Save model artifacts."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'params': {
                'n_lags': self.n_lags,
                'threshold_drop': self.threshold_drop
            }
        }
        joblib.dump(artifacts, path)
        
    @classmethod
    def load(cls, path: str):
        """Load model."""
        artifacts = joblib.load(path)
        
        instance = cls(
            n_lags=artifacts['params']['n_lags'],
            threshold_drop=artifacts['params']['threshold_drop']
        )
        instance.model = artifacts['model']
        instance.scaler = artifacts['scaler']
        instance.is_fitted = True
        return instance
