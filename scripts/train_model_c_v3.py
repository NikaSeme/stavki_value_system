#!/usr/bin/env python3
"""
Model C v3 — Transformer Architecture with Optuna Tuning
=========================================================

Features:
1. Transformer-based architecture for better feature interactions
2. Optuna hyperparameter optimization
3. Extended feature set (xG, CLV, O/U, H2H)
4. Multi-head attention for odds/stats feature groups
5. Calibration with Temperature Scaling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import json
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import sys

# Optional Optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available. Install with: pip install optuna")

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureGroup(nn.Module):
    """Process a group of related features."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single Transformer block with self-attention."""
    
    def __init__(self, d_model: int, n_heads: int = 4, dim_ff: int = 128, dropout: float = 0.2):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feedforward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class TransformerPredictor(nn.Module):
    """
    Transformer-based match outcome predictor.
    
    Architecture:
    1. Feature group embeddings (Elo, Form, Market, xG, etc.)
    2. Transformer blocks for feature interaction
    3. Classification head with temperature scaling
    """
    
    def __init__(
        self,
        feature_dims: Dict[str, int],
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.3,
        n_classes: int = 3,
    ):
        super().__init__()
        
        self.feature_dims = feature_dims
        self.d_model = d_model
        
        # Feature group embedders
        self.embedders = nn.ModuleDict()
        for name, dim in feature_dims.items():
            self.embedders[name] = FeatureGroup(dim, d_model * 2, d_model, dropout)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Classification head
        n_groups = len(feature_dims)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * n_groups, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, n_classes),
        )
        
        # Temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Embed each feature group
        embeddings = []
        for name in self.feature_dims.keys():
            if name in x:
                emb = self.embedders[name](x[name])
                embeddings.append(emb)
        
        # Stack as sequence: [batch, n_groups, d_model]
        x = torch.stack(embeddings, dim=1)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Flatten and classify
        x = x.flatten(start_dim=1)
        logits = self.classifier(x)
        
        # Temperature scaling
        logits = logits / self.temperature.clamp(min=0.1, max=10.0)
        
        return logits


class ModelCV3:
    """Model C v3 with Transformer architecture."""
    
    # Feature groups for the model
    FEATURE_GROUPS = {
        'elo': ['HomeEloBefore', 'AwayEloBefore', 'EloDiff', 'EloExpHome', 'EloExpAway'],
        'form': ['Home_Pts_L5', 'Home_GF_L5', 'Home_GA_L5', 'Away_Pts_L5', 'Away_GF_L5', 'Away_GA_L5'],
        'overall_form': ['Home_Overall_Pts_L5', 'Home_Overall_GF_L5', 'Home_Overall_GA_L5',
                         'Away_Overall_Pts_L5', 'Away_Overall_GF_L5', 'Away_Overall_GA_L5'],
        'market': ['Odds_Volatility', 'Market_Consensus', 'Sharp_Divergence',
                   'odds_implied_home', 'odds_implied_draw', 'odds_implied_away'],
        'xg': ['xG_Home_L5', 'xGA_Home_L5', 'xG_Away_L5', 'xGA_Away_L5', 'xG_Diff'],
        'clv': ['CLV_Home', 'CLV_Draw', 'CLV_Away'],
        'h2h': ['H2H_Home_Win_Pct', 'H2H_Goals_Avg', 'H2H_Matches'],
        'other': ['SentimentHome', 'SentimentAway', 'HomeInjury', 'AwayInjury',
                  'value_home', 'value_away'],
    }
    
    def __init__(self, params: Optional[Dict] = None):
        self.params = params or {
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 2,
            'dim_ff': 128,
            'dropout': 0.3,
            'lr': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 64,
            'epochs': 100,
            'patience': 30,
        }
        
        self.model = None
        self.scalers = {}
        self.calibrators = []
        self.feature_groups_actual = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def _prepare_feature_groups(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Filter feature groups to only include available columns."""
        available = set(df.columns)
        groups = {}
        
        for name, features in self.FEATURE_GROUPS.items():
            present = [f for f in features if f in available]
            if present:
                groups[name] = present
        
        return groups
    
    def _create_group_tensors(self, df: pd.DataFrame, fit: bool = False) -> Dict[str, torch.Tensor]:
        """Create scaled tensors for each feature group."""
        tensors = {}
        
        for name, features in self.feature_groups_actual.items():
            X = df[features].values.astype(np.float32)
            
            # Handle NaN
            X = np.nan_to_num(X, nan=0.0)
            
            if fit:
                self.scalers[name] = StandardScaler()
                X = self.scalers[name].fit_transform(X)
            else:
                if name in self.scalers:
                    X = self.scalers[name].transform(X)
            
            tensors[name] = torch.FloatTensor(X)
        
        return tensors
    
    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame, 
              y_train: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train the model."""
        
        # Prepare feature groups
        self.feature_groups_actual = self._prepare_feature_groups(df_train)
        feature_dims = {name: len(features) for name, features in self.feature_groups_actual.items()}
        
        logger.info(f"Feature groups: {list(feature_dims.keys())}")
        logger.info(f"Total features: {sum(feature_dims.values())}")
        
        # Create model
        self.model = TransformerPredictor(
            feature_dims=feature_dims,
            d_model=self.params['d_model'],
            n_heads=self.params['n_heads'],
            n_layers=self.params['n_layers'],
            dim_ff=self.params['dim_ff'],
            dropout=self.params['dropout'],
        ).to(self.device)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Prepare data
        X_train = self._create_group_tensors(df_train, fit=True)
        X_val = self._create_group_tensors(df_val, fit=False)
        
        y_train_t = torch.LongTensor(y_train)
        y_val_t = torch.LongTensor(y_val)
        
        # Class weights
        class_counts = np.bincount(y_train, minlength=3)
        class_weights = len(y_train) / (3 * class_counts + 1e-6)
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.params['lr'],
            weight_decay=self.params['weight_decay'],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=self.params['lr'] * 0.01
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(self.params['epochs']):
            # Train
            self.model.train()
            train_loss = 0.0
            
            # Mini-batch
            n = len(y_train)
            indices = np.random.permutation(n)
            batch_size = self.params['batch_size']
            
            for i in range(0, n, batch_size):
                batch_idx = indices[i:i+batch_size]
                
                batch_X = {name: X_train[name][batch_idx].to(self.device) 
                          for name in X_train}
                batch_y = y_train_t[batch_idx].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= (n // batch_size + 1)
            scheduler.step()
            
            # Validate
            self.model.eval()
            with torch.no_grad():
                val_X = {name: X_val[name].to(self.device) for name in X_val}
                val_outputs = self.model(val_X)
                val_loss = criterion(val_outputs, y_val_t.to(self.device)).item()
                val_pred = val_outputs.argmax(dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, val_pred)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or epoch < 5:
                logger.info(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, "
                           f"val_loss={val_loss:.4f}, val_acc={val_acc:.2%}")
            
            if patience_counter >= self.params['patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        self.model.load_state_dict(best_state)
        
        return {'history': history, 'best_val_loss': best_val_loss}
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        self.model.eval()
        X = self._create_group_tensors(df, fit=False)
        
        with torch.no_grad():
            X_device = {name: X[name].to(self.device) for name in X}
            logits = self.model(X_device)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        # Apply calibration if available
        if self.calibrators:
            calibrated = np.zeros_like(probs)
            for i, cal in enumerate(self.calibrators):
                calibrated[:, i] = cal.predict(probs[:, i])
            probs = calibrated / calibrated.sum(axis=1, keepdims=True)
        
        return probs
    
    def fit_calibration(self, df: pd.DataFrame, y: np.ndarray):
        """Fit isotonic calibrators."""
        probs = self.predict_proba(df)
        
        self.calibrators = []
        for i in range(3):
            cal = IsotonicRegression(out_of_bounds='clip')
            cal.fit(probs[:, i], (y == i).astype(int))
            self.calibrators.append(cal)
        
        logger.info("Calibrators fitted")


def run_optuna_study(df: pd.DataFrame, y: np.ndarray, n_trials: int = 50):
    """Run Optuna hyperparameter optimization."""
    
    if not OPTUNA_AVAILABLE:
        logger.error("Optuna not available. Install with: pip install optuna")
        return None
    
    def objective(trial):
        params = {
            'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),
            'n_heads': trial.suggest_categorical('n_heads', [2, 4, 8]),
            'n_layers': trial.suggest_int('n_layers', 1, 4),
            'dim_ff': trial.suggest_categorical('dim_ff', [64, 128, 256]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'epochs': 50,
            'patience': 15,
        }
        
        # TimeSeriesCV
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(df):
            df_train = df.iloc[train_idx]
            df_val = df.iloc[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]
            
            model = ModelCV3(params)
            try:
                result = model.train(df_train, df_val, y_train, y_val)
                scores.append(result['best_val_loss'])
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('inf')
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best params: {study.best_params}")
    logger.info(f"Best value: {study.best_value:.4f}")
    
    return study.best_params


def main():
    logger.info("=" * 70)
    logger.info("MODEL C v3 — TRANSFORMER TRAINING")
    logger.info("=" * 70)
    
    base_dir = Path(__file__).parent.parent
    
    # First run enrichment if needed
    enriched_file = base_dir / 'data' / 'processed' / 'multi_league_enriched.csv'
    if not enriched_file.exists():
        logger.info("Running data enrichment first...")
        import subprocess
        subprocess.run([sys.executable, str(base_dir / 'scripts' / 'enrich_data.py')], check=True)
    
    # Load enriched data
    df = pd.read_csv(enriched_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    y = df['FTR'].map({'H': 0, 'D': 1, 'A': 2}).values
    
    logger.info(f"Dataset: {len(df)} samples")
    
    # Split
    n = len(df)
    train_end = int(n * 0.70)
    cal_end = int(n * 0.80)
    
    df_train = df.iloc[:train_end]
    df_cal = df.iloc[train_end:cal_end]
    df_test = df.iloc[cal_end:]
    
    y_train = y[:train_end]
    y_cal = y[train_end:cal_end]
    y_test = y[cal_end:]
    
    logger.info(f"Split: Train={len(df_train)}, Cal={len(df_cal)}, Test={len(df_test)}")
    
    # Optuna tuning (if available and requested)
    best_params = None
    if OPTUNA_AVAILABLE and len(sys.argv) > 1 and sys.argv[1] == '--tune':
        logger.info("\nRunning Optuna optimization...")
        best_params = run_optuna_study(df.iloc[:train_end], y[:train_end], n_trials=30)
    
    # Train with best or default params
    model = ModelCV3(best_params)
    result = model.train(df_train, df_cal, y_train, y_cal)
    
    # Calibration
    model.fit_calibration(df_cal, y_cal)
    
    # Evaluate
    probs = model.predict_proba(df_test)
    y_pred = probs.argmax(axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, np.clip(probs, 1e-6, 1-1e-6))
    
    # Brier
    brier = np.mean([brier_score_loss((y_test == i).astype(int), probs[:, i]) for i in range(3)])
    
    logger.info("\n" + "=" * 50)
    logger.info("RESULTS")
    logger.info("=" * 50)
    logger.info(f"Test Accuracy: {accuracy:.2%}")
    logger.info(f"Test LogLoss:  {logloss:.4f}")
    logger.info(f"Test Brier:    {brier:.4f}")
    
    # Calibration check
    logger.info("\nCalibration:")
    for i, name in enumerate(['Home', 'Draw', 'Away']):
        pred = probs[:, i].mean()
        actual = (y_test == i).mean()
        logger.info(f"  {name}: pred={pred:.1%}, actual={actual:.1%}, gap={pred-actual:+.1%}")
    
    # Save model
    save_path = base_dir / 'models' / f'neural_v3_transformer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
    torch.save({
        'model_state': model.model.state_dict(),
        'scalers': model.scalers,
        'feature_groups': model.feature_groups_actual,
        'params': model.params,
    }, save_path)
    logger.info(f"\nModel saved to {save_path}")


if __name__ == '__main__':
    main()
