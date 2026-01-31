"""
Model C: Dense Neural Network for match prediction.

Simpler than LSTM - uses same 22 features as CatBoost.
Focus: Complement CatBoost with different learning approach.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
import pickle
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatchDataset(Dataset):
    """PyTorch dataset for match prediction."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DenseNN(nn.Module):
    """
    Dense neural network for match outcome prediction.
    
    Architecture:
    - Input: 22 features (same as CatBoost)
    - Hidden: 64 -> 32 -> 16
    - Output: 3 classes (Home/Draw/Away)
    - Regularization: Dropout, BatchNorm
    """
    
    def __init__(self, input_dim=22, hidden_dims=[64, 32, 16], dropout=0.3):
        super(DenseNN, self).__init__()
        
        # Input layer
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.drop1 = nn.Dropout(dropout)
        
        # Hidden layer 1
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.drop2 = nn.Dropout(dropout)
        
        # Hidden layer 2
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.drop3 = nn.Dropout(dropout * 0.5)  # Less dropout in final layer
        
        # Output layer
        self.fc4 = nn.Linear(hidden_dims[2], 3)
        
        # Activation
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.drop3(x)
        
        # Output
        x = self.fc4(x)
        
        return x  # Return logits (softmax applied later)


class NeuralModelTrainer:
    """Training and evaluation for Model C."""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.scaler = StandardScaler()
        self.calibrators = []
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, patience=10):
        """Train model with early stopping."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        logger.info(f"Training for {epochs} epochs (early stopping patience={patience})")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = correct / total
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")
        else:
            logger.warning("Training completed without improvement (NaN loss?). Using last state.")
    
    def predict_proba(self, X):
        """Predict probabilities."""
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
        
        return probs.cpu().numpy()
    
    def calibrate(self, X_cal, y_cal):
        """Apply isotonic calibration."""
        logger.info("Calibrating model...")
        
        # Get uncalibrated probabilities
        probs_uncal = self.predict_proba(X_cal)
        
        # Fit calibrator for each class
        self.calibrators = []
        for i in range(3):
            y_binary = (y_cal == i).astype(int)
            cal = IsotonicRegression(out_of_bounds='clip')
            cal.fit(probs_uncal[:, i], y_binary)
            self.calibrators.append(cal)
        
        logger.info("Calibration complete")
    
    def predict_calibrated(self, X):
        """Predict with calibration."""
        probs = self.predict_proba(X)
        
        if self.calibrators:
            calibrated = np.zeros_like(probs)
            for i, cal in enumerate(self.calibrators):
                calibrated[:, i] = cal.predict(probs[:, i])
            
            # Renormalize
            row_sums = calibrated.sum(axis=1, keepdims=True)
            calibrated = calibrated / row_sums
            
            return calibrated
        
        return probs
    
    def evaluate(self, X, y, calibrated=True):
        """Evaluate model."""
        if calibrated and self.calibrators:
            probs = self.predict_calibrated(X)
        else:
            probs = self.predict_proba(X)
        
        # Predictions
        y_pred = probs.argmax(axis=1)
        
        # Metrics
        acc = accuracy_score(y, y_pred)
        logloss = log_loss(y, probs)
        
        # Brier per class
        brier_scores = []
        for i in range(3):
            y_binary = (y == i).astype(int)
            brier = brier_score_loss(y_binary, probs[:, i])
            brier_scores.append(brier)
        
        return {
            'accuracy': acc,
            'log_loss': logloss,
            'brier_score': np.mean(brier_scores),
            'brier_per_class': brier_scores
        }
    
    def save(self, filepath):
        """Save model and calibrators."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'calibrators': self.calibrators,
            'history': self.history
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, model_arch):
        """Load model."""
        checkpoint = torch.load(filepath)
        
        trainer = cls(model_arch)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.scaler = checkpoint['scaler']
        trainer.calibrators = checkpoint['calibrators']
        trainer.history = checkpoint.get('history', {})
        
        logger.info(f"Model loaded from {filepath}")
        return trainer


def main():
    """Train Model C."""
    logger.info("=" * 70)
    logger.info("MODEL C (NEURAL NETWORK) TRAINING")
    logger.info("=" * 70)
    
    # Load data
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / 'data' / 'processed' / 'multi_league_features_2021_2024.csv'
    
    if not data_file.exists():
        logger.error(f"Data missing: {data_file}")
        return
    
    logger.info(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Features and labels
    # CRITICAL: Exclude match outcomes (Leakage) and non-numeric cols
    exclude_cols = ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'FTR', 'League',
                    'FTHG', 'FTAG', 'GoalDiff', 'TotalGoals', 'index',
                    'HomeEloAfter', 'AwayEloAfter'] # Prevent ELO Leakage!
    
    # 1. Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    # 2. Filter exclusions
    feature_cols = [col for col in numeric_df.columns if col not in exclude_cols]
    
    # Keep X as DataFrame to preserve feature names for Scaler
    X = df[feature_cols].copy()
    
    result_map = {'H': 2, 'D': 1, 'A': 0}
    y = df['FTR'].map(result_map).values
    
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Samples: {len(X)}")
    
    # 1. Handle NaNs in data (Pandas way)
    if X.isna().any().any():
        logger.warning(f"Found NaNs in input data. Filling with 0.")
        X = X.fillna(0.0)
        
    # 2. Drop Constant Features (causes StandardScaler NaN)
    # Check standard deviation of columns
    std = X.std()
    constant_cols = std[std < 1e-6].index.tolist()
    if constant_cols:
         logger.warning(f"Dropping {len(constant_cols)} constant features: {constant_cols}")
         X = X.drop(columns=constant_cols)
         feature_cols = X.columns.tolist()
         logger.info(f"Remaining Features: {len(feature_cols)}")

    # Split (same as other models)
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train, y_train = X.iloc[:train_end], y[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y[val_end:]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
    except Exception as e:
        logger.error(f"Scaling failed: {e}")
        return
    
    # Check for NaNs after scaling
    if np.isnan(X_train_scaled).any():
         logger.error("NaNs produced during scaling! Aborting.")
         return

    # Create datasets
    train_dataset = MatchDataset(X_train_scaled, y_train)
    val_dataset = MatchDataset(X_val_scaled, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = DenseNN(input_dim=len(feature_cols), hidden_dims=[64, 32, 16], dropout=0.3)
    trainer = NeuralModelTrainer(model)
    trainer.scaler = scaler
    
    # Train
    logger.info("\nTraining neural network...")
    trainer.train(train_loader, val_loader, epochs=200, lr=0.001, patience=15)
    
    # Calibrate
    trainer.calibrate(X_val_scaled, y_val)
    
    # Evaluate
    logger.info("\nEvaluation Results:")
    
    for name, X_set, y_set in [('Train', X_train_scaled, y_train), 
                                 ('Val', X_val_scaled, y_val),
                                 ('Test', X_test_scaled, y_test)]:
        metrics = trainer.evaluate(X_set, y_set, calibrated=True)
        logger.info(f"\n{name} Set:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Brier: {metrics['brier_score']:.4f}")
        logger.info(f"  LogLoss: {metrics['log_loss']:.4f}")
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = base_dir / 'models'
    model_file = model_dir / f'neural_v1_{timestamp}.pt'
    model_latest = model_dir / 'neural_v1_latest.pt'
    
    trainer.save(model_file)
    
    # Symlink
    if model_latest.exists():
        model_latest.unlink()
    model_latest.symlink_to(model_file.name)
    
    # Save metadata
    test_metrics = trainer.evaluate(X_test_scaled, y_test, calibrated=True)
    
    metadata = {
        'model': 'DenseNN',
        'version': 'v1',
        'train_date': timestamp,
        'architecture': {
            'input_dim': len(feature_cols),
            'hidden_dims': [64, 32, 16],
            'dropout': 0.3
        },
        'training': {
            'epochs_total': len(trainer.history['train_loss']),
            'final_train_loss': trainer.history['train_loss'][-1],
            'final_val_loss': trainer.history['val_loss'][-1],
            'best_val_acc': max(trainer.history['val_acc'])
        },
        'metrics': {
            'test': {
                'accuracy': float(test_metrics['accuracy']),
                'brier_score': float(test_metrics['brier_score']),
                'log_loss': float(test_metrics['log_loss'])
            }
        }
    }
    
    meta_file = model_dir / f'neural_metadata_{timestamp}.json'
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\n{'=' * 70}")
    logger.info("✅ MODEL C TRAINING COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"Model: {model_file}")
    logger.info(f"Metadata: {meta_file}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.2%}")
    logger.info(f"Test Brier: {test_metrics['brier_score']:.4f}")


if __name__ == '__main__':
    main()
