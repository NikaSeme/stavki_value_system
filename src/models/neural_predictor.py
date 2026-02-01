"""
Neural model predictor for live predictions.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DenseNN(nn.Module):
    """Dense neural network architecture (embedded copy)."""
    
    def __init__(self, input_dim=22, hidden_dims=[64, 32, 16], dropout=0.3):
        super(DenseNN, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.drop1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.drop2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.drop3 = nn.Dropout(dropout * 0.5)
        
        self.fc4 = nn.Linear(hidden_dims[2], 3)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.drop3(x)
        
        x = self.fc4(x)
        return x


class NeuralPredictor:
    """Wrapper for Model C (neural network) predictions."""
    
    def __init__(self, model_file='models/neural_v1_latest.pt'):
        """Load neural model."""
        self.model_file = Path(model_file)
        self.model = None
        self.scaler = None
        self.calibrators = None
        self.feature_columns = None
        self._load()
    
    def _load(self):
        """Load model from file."""
        logger.info(f"Loading neural model from {self.model_file}")
        
        # Load feature contract
        contract_file = self.model_file.parent / 'neural_feature_columns.json'
        if contract_file.exists():
            import json
            with open(contract_file, 'r') as f:
                self.feature_columns = json.load(f)
            logger.info(f"Loaded feature contract: {len(self.feature_columns)} features")
        else:
            logger.warning("No feature contract found! Inference may be unstable.")
        
        # Load checkpoint
        # weights_only=False is required because the checkpoint contains a pickled StandardScaler
        try:
            checkpoint = torch.load(self.model_file, weights_only=False)
        except TypeError:
             # Fallback for older torch versions lacking weights_only arg
            checkpoint = torch.load(self.model_file)
        
        # Create model
        # DYNAMICALLY determine input dim from checkpoint weight shape
        state_dict = checkpoint['model_state_dict']
        input_dim = state_dict['fc1.weight'].shape[1]
        logger.info(f"Dynamic Input Dim: {input_dim}")
        
        self.model = DenseNN(input_dim=input_dim, hidden_dims=[64, 32, 16], dropout=0.3)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Load scaler and calibrators
        self.scaler = checkpoint['scaler']
        self.calibrators = checkpoint['calibrators']
        
        logger.info("✓ Neural model loaded")

    def get_feature_names(self):
        """Get list of expected features."""
        if self.feature_columns:
            return self.feature_columns
        # Fallback to scaler features if available
        if self.scaler and hasattr(self.scaler, 'feature_names_in_'):
            return list(self.scaler.feature_names_in_)
        return None
    
    def predict(self, X):
        """
        Predict probabilities.
        
        Args:
            X: Feature array or DataFrame
            
        Returns:
            Probabilities (n_samples, 3) - [Away, Draw, Home]
        """
        # Data Safety: Enforce Feature Contract
        if hasattr(X, 'columns'):
            if self.feature_columns:
                # 1. Enforce strict column list (reorder + select)
                X_aligned = pd.DataFrame(index=X.index)
                
                missing = []
                for col in self.feature_columns:
                    if col in X.columns:
                        X_aligned[col] = X[col]
                    else:
                        X_aligned[col] = 0.0
                        missing.append(col)
                
                if missing:
                    logger.debug(f"Neural Model missing inputs (filled 0.0): {missing}")
                
                # Update X to aligned version
                X = X_aligned
            else:
                # Fallback Logic (Old)
                try:
                    if hasattr(self.scaler, 'feature_names_in_'):
                        X = X[self.scaler.feature_names_in_]
                    else:
                        X = X.select_dtypes(include=[np.number])
                except KeyError as e:
                    logger.error(f"Missing columns for Neural Model: {e}")
                    X = X.select_dtypes(include=[np.number])
        
        # Ensure we have a dataframe or array with correct shape
        if self.feature_columns and hasattr(X, 'shape'):
             if X.shape[1] != len(self.feature_columns):
                 logger.warning(f"Shape mismatch: Input {X.shape[1]} != Contract {len(self.feature_columns)}")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        X_tensor = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1).numpy()
        
        # Apply calibration
        if self.calibrators:
            calibrated = np.zeros_like(probs)
            for i, cal in enumerate(self.calibrators):
                calibrated[:, i] = cal.predict(probs[:, i])
            
            # Renormalize
            row_sums = calibrated.sum(axis=1, keepdims=True)
            probs = calibrated / row_sums
        
        return probs
