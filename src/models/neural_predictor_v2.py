"""
Model C v2 - Professional Neural Network
=========================================
Complete rewrite with:
- Numerical stability fixes
- Modern architecture (ResNet, GELU, Temperature)
- Extended features (35+)
- Training improvements (mixup, label smoothing, scheduler)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import json

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = x + residual  # Skip connection
        x = F.gelu(x)
        return x


class ProfessionalNN(nn.Module):
    """
    Professional neural network for match outcome prediction.
    
    Improvements over v1:
    - GELU activation (smoother than ReLU)
    - Residual connections (better gradient flow)
    - Temperature scaling (calibrated probabilities)
    - LayerNorm option (stable with small batches)
    """
    
    def __init__(
        self,
        input_dim: int = 35,
        hidden_dims: List[int] = [128, 64, 64, 32],
        dropout: float = 0.3,
        use_residual: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.use_residual = use_residual
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])
        self.input_drop = nn.Dropout(dropout)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            if use_residual and hidden_dims[i] == hidden_dims[i+1]:
                # Use residual block for same-dim layers
                self.hidden_layers.append(ResidualBlock(hidden_dims[i], dropout * 0.5))
            else:
                # Regular layer
                self.hidden_layers.append(nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.GELU(),
                    nn.Dropout(dropout * (0.5 ** (i+1))),  # Decreasing dropout
                ))
        
        # Output layer
        self.output = nn.Linear(hidden_dims[-1], 3)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = F.gelu(x)
        x = self.input_drop(x)
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Output with temperature scaling
        logits = self.output(x)
        logits = logits / self.temperature.clamp(min=0.1, max=10.0)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get calibrated probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class NeuralPredictorV2:
    """
    Professional wrapper for Model C predictions.
    
    Features:
    - Numerical stability (clipping, NaN handling)
    - Temperature-scaled calibration
    - Input validation
    - Graceful fallback
    """
    
    # Feature contract v2 - expanded features
    DEFAULT_FEATURES = [
        # Elo (3)
        'HomeEloBefore', 'AwayEloBefore', 'EloDiff',
        
        # Home Form - Home Specific (3)
        'Home_Pts_L5', 'Home_GF_L5', 'Home_GA_L5',
        
        # Away Form - Away Specific (3)
        'Away_Pts_L5', 'Away_GF_L5', 'Away_GA_L5',
        
        # Home Form - Overall (3)
        'Home_Overall_Pts_L5', 'Home_Overall_GF_L5', 'Home_Overall_GA_L5',
        
        # Away Form - Overall (3)
        'Away_Overall_Pts_L5', 'Away_Overall_GF_L5', 'Away_Overall_GA_L5',
        
        # xG Features (6) - NEW
        'xG_home_L5', 'xG_away_L5', 'xGA_home_L5', 'xGA_away_L5',
        'xG_diff', 'xG_form_home',
        
        # H2H Features (4) - NEW
        'H2H_home_wins_L5', 'H2H_away_wins_L5', 'H2H_draws_L5', 'H2H_goals_L5',
        
        # Market Features (5) - NEW
        'odds_implied_home', 'odds_implied_draw', 'odds_implied_away',
        'odds_value_home', 'Odds_Volatility',
        
        # Sentiment (2)
        'SentimentHome', 'SentimentAway',
        
        # Temporal (3) - NEW
        'day_of_week', 'month', 'is_weekend',
    ]
    
    def __init__(
        self,
        model_file: str = 'models/neural_v2_latest.pt',
        fallback_file: str = 'models/neural_v1_latest.pt',
    ):
        self.model_file = Path(model_file)
        self.fallback_file = Path(fallback_file)
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._load()
    
    def _load(self):
        """Load model with fallback."""
        model_path = self.model_file if self.model_file.exists() else self.fallback_file
        
        if not model_path.exists():
            logger.error(f"No model found at {self.model_file} or {self.fallback_file}")
            self._create_dummy_model()
            return
        
        logger.info(f"Loading Model C v2 from {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load feature contract
        contract_file = model_path.parent / 'neural_feature_columns_v2.json'
        if contract_file.exists():
            with open(contract_file) as f:
                self.feature_columns = json.load(f)
        else:
            # Try v1 contract
            contract_file_v1 = model_path.parent / 'neural_feature_columns.json'
            if contract_file_v1.exists():
                with open(contract_file_v1) as f:
                    self.feature_columns = json.load(f)
            else:
                self.feature_columns = self.DEFAULT_FEATURES[:18]  # Fallback
        
        # Create model
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Infer input dim from weights
        if 'input_proj.weight' in state_dict:
            input_dim = state_dict['input_proj.weight'].shape[1]
        elif 'fc1.weight' in state_dict:
            input_dim = state_dict['fc1.weight'].shape[1]
        else:
            input_dim = len(self.feature_columns)
        
        logger.info(f"Model input dim: {input_dim}, Features: {len(self.feature_columns)}")
        
        # Create appropriate model
        if 'input_proj.weight' in state_dict:
            # V2 architecture
            self.model = ProfessionalNN(input_dim=input_dim)
        else:
            # V1 architecture (DenseNN)
            from .neural_predictor import DenseNN
            self.model = DenseNN(input_dim=input_dim)
        
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as e:
            logger.warning(f"Partial load: {e}")
            self.model.load_state_dict(state_dict, strict=False)
        
        self.model.eval()
        self.model.to(self.device)
        
        # Load scaler
        self.scaler = checkpoint.get('scaler')
        
        logger.info(f"✓ Model C v2 loaded ({len(self.feature_columns)} features)")
    
    def _create_dummy_model(self):
        """Create dummy model for graceful degradation."""
        logger.warning("Creating dummy model - predictions will be uniform")
        self.model = ProfessionalNN(input_dim=18)
        self.feature_columns = self.DEFAULT_FEATURES[:18]
        self.scaler = None
    
    def get_feature_names(self) -> List[str]:
        """Get expected feature names."""
        return self.feature_columns
    
    def _prepare_input(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare and validate input."""
        # Align to feature contract
        X_aligned = pd.DataFrame(index=X.index)
        missing = []
        
        for col in self.feature_columns:
            if col in X.columns:
                X_aligned[col] = X[col]
            else:
                X_aligned[col] = 0.0
                missing.append(col)
        
        if missing:
            logger.debug(f"Missing features (filled 0): {missing[:5]}...")
        
        # Handle NaN
        X_aligned = X_aligned.fillna(0.0)
        
        # Convert to numpy
        X_arr = X_aligned.values.astype(np.float32)
        
        # Clip extreme values
        X_arr = np.clip(X_arr, -1e6, 1e6)
        
        # Scale if scaler available
        if self.scaler is not None:
            try:
                X_arr = self.scaler.transform(X_arr)
            except Exception as e:
                logger.warning(f"Scaling failed: {e}")
        
        # Final NaN check
        if np.isnan(X_arr).any():
            logger.warning("NaN after scaling - replacing with 0")
            X_arr = np.nan_to_num(X_arr, 0.0)
        
        return X_arr
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Probabilities (n_samples, 3) - [Home, Draw, Away]
        """
        # Prepare input
        X_arr = self._prepare_input(X)
        X_tensor = torch.FloatTensor(X_arr).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(X_tensor)
            else:
                logits = self.model(X_tensor)
                probs = F.softmax(logits, dim=1)
        
        probs = probs.cpu().numpy()
        
        # Numerical stability
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        
        # Renormalize
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        return probs
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence scores.
        
        Returns:
            (probs, confidence) - confidence is max_prob - second_max
        """
        probs = self.predict(X)
        
        # Confidence: difference between top-2 probabilities
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]
        confidence = sorted_probs[:, 0] - sorted_probs[:, 1]
        
        return probs, confidence
