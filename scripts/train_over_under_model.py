#!/usr/bin/env python3
"""
Over/Under 2.5 Goals Model
===========================

Binary classifier for total goals > 2.5
Uses similar architecture to Model C but optimized for O/U market.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score
from sklearn.isotonic import IsotonicRegression
import json
from datetime import datetime
import logging
from typing import Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OverUnderNN(nn.Module):
    """Neural network for Over/Under 2.5 prediction."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32], dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class OverUnderModel:
    """Over/Under 2.5 Goals Model."""
    
    FEATURES = [
        # Form
        'Home_GF_L5', 'Home_GA_L5', 'Away_GF_L5', 'Away_GA_L5',
        'Home_Overall_GF_L5', 'Home_Overall_GA_L5',
        'Away_Overall_GF_L5', 'Away_Overall_GA_L5',
        # xG
        'xG_Home_L5', 'xGA_Home_L5', 'xG_Away_L5', 'xGA_Away_L5',
        # O/U specific
        'OU_Home_Pct_L5', 'OU_Away_Pct_L5', 'Total_Goals_Exp',
        # H2H
        'H2H_Goals_Avg',
        # Elo
        'EloDiff', 'EloExpHome', 'EloExpAway',
        # Market (if O/U odds available)
        'Odds_Volatility',
    ]
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.calibrator = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.threshold = 0.5
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix."""
        available = [f for f in self.FEATURES if f in df.columns]
        X = df[available].fillna(0).values.astype(np.float32)
        return X, available
    
    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
              epochs: int = 100, patience: int = 20, lr: float = 0.001):
        """Train the model."""
        
        # Prepare features
        X_train, features = self._prepare_features(df_train)
        X_val, _ = self._prepare_features(df_val)
        
        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        # Targets
        y_train = ((df_train['FTHG'] + df_train['FTAG']) > 2.5).astype(np.float32).values
        y_val = ((df_val['FTHG'] + df_val['FTAG']) > 2.5).astype(np.float32).values
        
        logger.info(f"Features: {len(features)}")
        logger.info(f"Train O/U distribution: Over={y_train.mean():.1%}, Under={1-y_train.mean():.1%}")
        
        # Create model
        self.model = OverUnderNN(len(features)).to(self.device)
        
        # Class weight for imbalance
        pos_weight = torch.tensor([(1 - y_train.mean()) / (y_train.mean() + 1e-6)]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Data
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=64, shuffle=True
        )
        
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)
        
        # Training
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            scheduler.step()
            
            # Validate
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                val_probs = torch.sigmoid(val_outputs).cpu().numpy()
                val_preds = (val_probs > 0.5).astype(int)
                val_acc = accuracy_score(y_val, val_preds)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_acc={val_acc:.2%}")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        self.model.load_state_dict(best_state)
        return {'best_val_loss': best_val_loss}
    
    def fit_calibration(self, df: pd.DataFrame):
        """Fit calibration on hold-out set."""
        X, _ = self._prepare_features(df)
        X = self.scaler.transform(X)
        y = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int).values
        
        probs = self.predict_proba(df)
        
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(probs, y)
        
        logger.info("O/U calibrator fitted")
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability of Over 2.5."""
        X, _ = self._prepare_features(df)
        X = self.scaler.transform(X)
        
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        if self.calibrator:
            probs = self.calibrator.predict(probs)
        
        return probs
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict Over (1) or Under (0)."""
        probs = self.predict_proba(df)
        return (probs > self.threshold).astype(int)


def calculate_ou_roi(probs: np.ndarray, y_true: np.ndarray, 
                     odds_over: np.ndarray, odds_under: np.ndarray,
                     threshold: float = 0.5) -> Dict:
    """Calculate ROI for O/U betting."""
    
    n = len(probs)
    total_stake = 0
    total_return = 0
    wins = 0
    losses = 0
    
    for i in range(n):
        # Value bet: when our prob differs from implied
        implied_over = 1 / odds_over[i] if odds_over[i] > 1 else 0.5
        
        if probs[i] > implied_over + 0.05:  # Value on Over
            stake = 1.0
            total_stake += stake
            if y_true[i] == 1:  # Over
                total_return += stake * odds_over[i]
                wins += 1
            else:
                losses += 1
        elif probs[i] < implied_over - 0.05:  # Value on Under
            stake = 1.0
            total_stake += stake
            if y_true[i] == 0:  # Under
                total_return += stake * odds_under[i]
                wins += 1
            else:
                losses += 1
    
    if total_stake == 0:
        return {'roi': 0, 'bets': 0, 'wins': 0, 'win_rate': 0}
    
    profit = total_return - total_stake
    
    return {
        'roi': profit / total_stake * 100,
        'bets': wins + losses,
        'wins': wins,
        'win_rate': wins / (wins + losses) * 100,
    }


def main():
    logger.info("=" * 70)
    logger.info("OVER/UNDER 2.5 MODEL TRAINING")
    logger.info("=" * 70)
    
    base_dir = Path(__file__).parent.parent
    
    # Load enriched data
    enriched_file = base_dir / 'data' / 'processed' / 'multi_league_enriched.csv'
    if not enriched_file.exists():
        logger.info("Running data enrichment first...")
        import subprocess
        subprocess.run([sys.executable, str(base_dir / 'scripts' / 'enrich_data.py')], check=True)
    
    df = pd.read_csv(enriched_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Target
    df['OU_Target'] = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)
    
    logger.info(f"Dataset: {len(df)} samples")
    logger.info(f"O/U Distribution: Over={df['OU_Target'].mean():.1%}, Under={1-df['OU_Target'].mean():.1%}")
    
    # Split
    n = len(df)
    train_end = int(n * 0.70)
    cal_end = int(n * 0.80)
    
    df_train = df.iloc[:train_end]
    df_cal = df.iloc[train_end:cal_end]
    df_test = df.iloc[cal_end:]
    
    # Train
    model = OverUnderModel()
    model.train(df_train, df_cal, epochs=100, patience=25)
    
    # Calibrate
    model.fit_calibration(df_cal)
    
    # Evaluate
    y_test = df_test['OU_Target'].values
    probs = model.predict_proba(df_test)
    preds = model.predict(df_test)
    
    accuracy = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    f1 = f1_score(y_test, preds)
    
    logger.info("\n" + "=" * 50)
    logger.info("RESULTS")
    logger.info("=" * 50)
    logger.info(f"Test Accuracy: {accuracy:.2%}")
    logger.info(f"Test AUC:      {auc:.4f}")
    logger.info(f"Test F1:       {f1:.4f}")
    
    # Calibration
    logger.info("\nCalibration:")
    over_pred = probs.mean()
    over_actual = y_test.mean()
    logger.info(f"  Predicted Over: {over_pred:.1%}")
    logger.info(f"  Actual Over:    {over_actual:.1%}")
    logger.info(f"  Gap:            {over_pred - over_actual:+.1%}")
    
    # ROI (simulated with 1.9 odds for both)
    logger.info("\nROI Analysis (simulated odds 1.9/1.9):")
    roi = calculate_ou_roi(probs, y_test, 
                          np.full(len(y_test), 1.9),
                          np.full(len(y_test), 1.9))
    logger.info(f"  Bets: {roi['bets']}")
    logger.info(f"  Win Rate: {roi['win_rate']:.0f}%")
    logger.info(f"  ROI: {roi['roi']:+.1f}%")
    
    # Save
    save_path = base_dir / 'models' / f'over_under_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
    torch.save({
        'model_state': model.model.state_dict(),
        'scaler': model.scaler,
    }, save_path)
    logger.info(f"\nModel saved to {save_path}")


if __name__ == '__main__':
    main()
