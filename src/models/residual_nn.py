"""
Residual Neural Network for Market Probability Adjustment.

Instead of predicting absolute probabilities, this model predicts
*adjustments* (deltas) to market probabilities.

Input: [market_probs (3), features (18)] = 21 features
Output: [delta_H, delta_D, delta_A] constrained to sum to 0

Final prediction: market_prob + model_delta
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.activation(x + self.layers(x))


class ResidualNN(nn.Module):
    """
    Neural network that predicts adjustments to market probabilities.
    
    Architecture:
    - Input: market_probs (3) + features (18) = 21
    - Hidden: 64 -> 32 -> 16 with residual connections
    - Output: 3 deltas (centered to sum to 0)
    """
    
    def __init__(self, input_dim=21, hidden_dims=[64, 32, 16], dropout=0.3):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Hidden layers
        self.hidden1 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.hidden2 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Output layer (3 deltas)
        self.output = nn.Linear(hidden_dims[2], 3)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, 21] = [market_probs(3), features(18)]
            
        Returns:
            Deltas [batch, 3] centered to sum to 0
        """
        x = self.input_proj(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        
        # Raw deltas
        deltas = self.output(x)
        
        # Center to sum to 0 (constraint)
        deltas = deltas - deltas.mean(dim=1, keepdim=True)
        
        return deltas


class ResidualPredictor:
    """
    Wrapper for ResidualNN inference.
    
    Loads model and applies deltas to market probabilities.
    """
    
    def __init__(self, model_file: str = 'models/residual_nn_latest.pt'):
        self.model_file = Path(model_file)
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self._load()
    
    def _load(self):
        """Load model from checkpoint."""
        if not self.model_file.exists():
            raise FileNotFoundError(f"Model not found: {self.model_file}")
        
        logger.info(f"Loading residual model from {self.model_file}")
        
        checkpoint = torch.load(self.model_file, map_location='cpu', weights_only=False)
        
        # Architecture params
        input_dim = checkpoint.get('input_dim', 21)
        hidden_dims = checkpoint.get('hidden_dims', [64, 32, 16])
        
        self.model = ResidualNN(input_dim=input_dim, hidden_dims=hidden_dims)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.scaler = checkpoint.get('scaler')
        self.feature_columns = checkpoint.get('feature_columns', [])
        
        logger.info(f"✓ Residual model loaded ({len(self.feature_columns)} features)")
    
    def predict(self, market_probs: np.ndarray, features: pd.DataFrame) -> np.ndarray:
        """
        Predict adjusted probabilities.
        
        Args:
            market_probs: [n_samples, 3] market probabilities (H, D, A)
            features: DataFrame with input features (18 columns)
            
        Returns:
            Adjusted probabilities [n_samples, 3]
        """
        # Align features to contract
        X_feat = self._align_features(features)
        
        # Scale features (not market probs)
        if self.scaler:
            X_feat_scaled = self.scaler.transform(X_feat)
        else:
            X_feat_scaled = X_feat.values
        
        # Combine: [market_probs, scaled_features]
        X_combined = np.hstack([market_probs, X_feat_scaled])
        
        # Predict deltas
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_combined)
            deltas = self.model(X_tensor).numpy()
        
        # Apply deltas to market probs
        adjusted = market_probs + deltas
        
        # Clip to valid range and renormalize
        adjusted = np.clip(adjusted, 0.01, 0.99)
        adjusted = adjusted / adjusted.sum(axis=1, keepdim=True)
        
        return adjusted
    
    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Align input features to training contract."""
        X_aligned = pd.DataFrame(index=features.index)
        
        for col in self.feature_columns:
            if col in features.columns:
                X_aligned[col] = features[col]
            else:
                X_aligned[col] = 0.0
        
        return X_aligned
    
    def get_feature_names(self):
        """Get expected feature column names."""
        return self.feature_columns
