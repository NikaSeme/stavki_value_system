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
        self._load()
    
    def _load(self):
        """Load model from file."""
        logger.info(f"Loading neural model from {self.model_file}")
        
        # Load checkpoint
        # weights_only=False is required because the checkpoint contains a pickled StandardScaler
        try:
            checkpoint = torch.load(self.model_file, weights_only=False)
        except TypeError:
             # Fallback for older torch versions lacking weights_only arg
            checkpoint = torch.load(self.model_file)
        
        # Create model
        self.model = DenseNN(input_dim=22, hidden_dims=[64, 32, 16], dropout=0.3)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load scaler and calibrators
        self.scaler = checkpoint['scaler']
        self.calibrators = checkpoint['calibrators']
        
        logger.info("✓ Neural model loaded")
    
    def predict(self, X):
        """
        Predict probabilities.
        
        Args:
            X: Feature array (n_samples, 22)
            
        Returns:
            Probabilities (n_samples, 3) - [Away, Draw, Home]
        """
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
