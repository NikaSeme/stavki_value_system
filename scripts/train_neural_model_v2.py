#!/usr/bin/env python3
"""
Model C v2 Training Script
==========================
Professional training with:
- Gradient clipping
- Learning rate scheduling  
- Label smoothing
- Mixup augmentation
- Extended features
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
import json
from datetime import datetime
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import model
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.neural_predictor_v2 import ProfessionalNN


class MatchDataset(Dataset):
    """Dataset with mixup support."""
    
    def __init__(self, X, y, mixup_alpha: float = 0.0):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.mixup_alpha = mixup_alpha
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        
        # Mixup during training
        if self.mixup_alpha > 0 and self.training:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            rand_idx = np.random.randint(len(self))
            x = lam * x + (1 - lam) * self.X[rand_idx]
            # For mixup, we return both labels
            return x, y, self.y[rand_idx], torch.tensor(lam)
        
        return x, y
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


class LabelSmoothingLoss(nn.Module):
    """Cross entropy with label smoothing."""
    
    def __init__(self, n_classes: int = 3, smoothing: float = 0.1):
        super().__init__()
        self.n_classes = n_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class ProfessionalTrainer:
    """Training with all improvements."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        label_smoothing: float = 0.1,
        gradient_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.scaler = StandardScaler()
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        
        self.label_smoothing = label_smoothing
        self.gradient_clip = gradient_clip
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 200,
        lr: float = 0.001,
        patience: int = 20,
    ):
        """Train with all improvements."""
        # Loss with label smoothing
        criterion = LabelSmoothingLoss(n_classes=3, smoothing=self.label_smoothing)
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )
        
        # Cosine annealing scheduler
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,  # Restart every 20 epochs
            T_mult=2,
            eta_min=lr * 0.01,
        )
        
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        
        logger.info(f"Training for {epochs} epochs (patience={patience})")
        logger.info(f"Label smoothing: {self.label_smoothing}, Gradient clip: {self.gradient_clip}")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                X_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            val_loss, val_acc = self._validate(val_loader, criterion)
            
            # Update scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Logging
            if epoch % 10 == 0 or patience_counter == 0:
                logger.info(
                    f"Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}, "
                    f"acc={val_acc:.4f}, lr={current_lr:.6f}"
                )
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)
            logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    
    def _validate(self, loader: DataLoader, criterion: nn.Module):
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in loader:
                X = batch[0].to(self.device)
                y = batch[1].to(self.device)
                
                outputs = self.model(X)
                loss = criterion(outputs, y)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        return total_loss / len(loader), correct / total
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probabilities."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
        
        return probs.cpu().numpy()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Comprehensive evaluation."""
        probs = self.predict_proba(X)
        y_pred = probs.argmax(axis=1)
        
        # Metrics
        acc = accuracy_score(y, y_pred)
        
        # Safe log_loss
        probs_clipped = np.clip(probs, 1e-6, 1 - 1e-6)
        probs_clipped = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)
        logloss = log_loss(y, probs_clipped)
        
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
            'brier_per_class': brier_scores,
        }
    
    def save(self, filepath: Path):
        """Save model checkpoint."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'history': self.history,
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")


def load_and_prepare_data(data_dir: Path) -> tuple:
    """Load data with extended features."""
    
    # Try multiple data sources
    data_files = [
        data_dir / 'processed' / 'multi_league_features_2021_2024.csv',
        data_dir / 'processed' / 'multi_league_features_peopled.csv',
        data_dir / 'ml_dataset_v2.csv',
    ]
    
    df = None
    for f in data_files:
        if f.exists():
            logger.info(f"Loading data from {f}")
            df = pd.read_csv(f)
            break
    
    if df is None:
        raise FileNotFoundError("No training data found!")
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date').dropna(subset=['Date'])
    
    # Extended feature list
    feature_cols = [
        # Elo
        'HomeEloBefore', 'AwayEloBefore', 'EloDiff',
        
        # Home Form
        'Home_Pts_L5', 'Home_GF_L5', 'Home_GA_L5',
        
        # Away Form
        'Away_Pts_L5', 'Away_GF_L5', 'Away_GA_L5',
        
        # Overall Form
        'Home_Overall_Pts_L5', 'Home_Overall_GF_L5', 'Home_Overall_GA_L5',
        'Away_Overall_Pts_L5', 'Away_Overall_GF_L5', 'Away_Overall_GA_L5',
        
        # Market
        'Odds_Volatility',
        
        # Sentiment
        'SentimentHome', 'SentimentAway',
    ]
    
    # Add optional features if available
    optional_features = [
        'xG_home_L5', 'xG_away_L5', 'xG_diff',
        'H2H_home_wins_L5', 'H2H_total_goals_L5',
        'odds_implied_home', 'odds_implied_draw', 'odds_implied_away',
    ]
    
    for col in optional_features:
        if col in df.columns:
            feature_cols.append(col)
            logger.info(f"Added optional feature: {col}")
    
    # Filter to available features
    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    
    if missing:
        logger.warning(f"Missing features: {missing}")
    
    logger.info(f"Using {len(available)} features")
    
    # Prepare data
    X = df[available].copy()
    X = X.fillna(0.0)
    
    # Labels
    result_map = {'H': 0, 'D': 1, 'A': 2}
    y = df['FTR'].map(result_map).values
    
    return X, y, available


def main():
    parser = argparse.ArgumentParser(description='Train Model C v2')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("MODEL C V2 - PROFESSIONAL TRAINING")
    logger.info("=" * 70)
    
    base_dir = Path(__file__).parent.parent
    
    # Load data
    X, y, feature_cols = load_and_prepare_data(base_dir / 'data')
    
    logger.info(f"Dataset: {len(X)} samples, {len(feature_cols)} features")
    logger.info(f"Class distribution: H={sum(y==0)}, D={sum(y==1)}, A={sum(y==2)}")
    
    # Split
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train, y_train = X.iloc[:train_end], y[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y[val_end:]
    
    logger.info(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)
    
    # Check for NaN after scaling
    if np.isnan(X_train_sc).any():
        logger.warning("NaN in scaled data - replacing with 0")
        X_train_sc = np.nan_to_num(X_train_sc, 0.0)
        X_val_sc = np.nan_to_num(X_val_sc, 0.0)
        X_test_sc = np.nan_to_num(X_test_sc, 0.0)
    
    # Create loaders
    train_dataset = MatchDataset(X_train_sc, y_train)
    val_dataset = MatchDataset(X_val_sc, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = ProfessionalNN(
        input_dim=len(feature_cols),
        hidden_dims=[128, 64, 64, 32],
        dropout=0.3,
        use_residual=True,
    )
    
    logger.info(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = ProfessionalTrainer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        label_smoothing=args.label_smoothing,
        gradient_clip=args.gradient_clip,
    )
    trainer.scaler = scaler
    
    # Train
    trainer.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )
    
    # Evaluate
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    
    for name, X_set, y_set in [
        ('Train', X_train_sc, y_train),
        ('Val', X_val_sc, y_val),
        ('Test', X_test_sc, y_test),
    ]:
        metrics = trainer.evaluate(X_set, y_set)
        logger.info(f"\n{name}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Brier:    {metrics['brier_score']:.4f}")
        logger.info(f"  LogLoss:  {metrics['log_loss']:.4f}")
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = base_dir / 'models'
    
    model_file = model_dir / f'neural_v2_{timestamp}.pt'
    model_latest = model_dir / 'neural_v2_latest.pt'
    
    trainer.save(model_file)
    
    # Update symlink
    if model_latest.exists() or model_latest.is_symlink():
        model_latest.unlink()
    model_latest.symlink_to(model_file.name)
    
    # Save feature contract
    contract_file = model_dir / 'neural_feature_columns_v2.json'
    with open(contract_file, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    
    # Save metadata
    test_metrics = trainer.evaluate(X_test_sc, y_test)
    
    metadata = {
        'model': 'ProfessionalNN',
        'version': 'v2',
        'train_date': timestamp,
        'architecture': {
            'input_dim': len(feature_cols),
            'hidden_dims': [128, 64, 64, 32],
            'dropout': 0.3,
            'use_residual': True,
        },
        'training': {
            'epochs': len(trainer.history['train_loss']),
            'label_smoothing': args.label_smoothing,
            'gradient_clip': args.gradient_clip,
            'final_train_loss': trainer.history['train_loss'][-1],
            'final_val_loss': trainer.history['val_loss'][-1],
            'best_val_acc': max(trainer.history['val_acc']),
        },
        'metrics': {
            'test': {
                'accuracy': float(test_metrics['accuracy']),
                'brier_score': float(test_metrics['brier_score']),
                'log_loss': float(test_metrics['log_loss']),
            }
        },
        'features': feature_cols,
    }
    
    meta_file = model_dir / f'neural_v2_metadata_{timestamp}.json'
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ MODEL C V2 TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Model: {model_file}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.2%}")
    logger.info(f"Test Brier: {test_metrics['brier_score']:.4f}")
    logger.info(f"Test LogLoss: {test_metrics['log_loss']:.4f}")


if __name__ == '__main__':
    main()
