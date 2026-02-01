#!/usr/bin/env python3
"""
Train Residual Neural Network.

Trains a model that predicts adjustments (deltas) to market probabilities
instead of absolute match outcome probabilities.

Input: market_probs (3) + features (18) = 21 features
Output: deltas (3) constrained to sum to 0
Target: one_hot(actual_result) - market_probs

Loss: MSE on residuals
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json
import logging
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResidualNN(nn.Module):
    """Neural network predicting market probability adjustments."""
    
    def __init__(self, input_dim=21, hidden_dims=[64, 32, 16], dropout=0.3):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
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
        
        self.output = nn.Linear(hidden_dims[2], 3)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        deltas = self.output(x)
        # Center to sum to 0
        deltas = deltas - deltas.mean(dim=1, keepdim=True)
        return deltas


def main():
    logger.info("=" * 70)
    logger.info("RESIDUAL NEURAL NETWORK TRAINING")
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
    
    # Feature columns (same as standard neural model)
    feature_cols = [
        'HomeEloBefore', 'AwayEloBefore', 'EloDiff',
        'Home_Pts_L5', 'Home_GF_L5', 'Home_GA_L5',
        'Away_Pts_L5', 'Away_GF_L5', 'Away_GA_L5',
        'Home_Overall_Pts_L5', 'Home_Overall_GF_L5', 'Home_Overall_GA_L5',
        'Away_Overall_Pts_L5', 'Away_Overall_GF_L5', 'Away_Overall_GA_L5',
        'Odds_Volatility', 'SentimentHome', 'SentimentAway'
    ]
    
    # Market probability columns (from historical odds)
    market_prob_cols = ['MarketProbHomeNoVig', 'MarketProbDrawNoVig', 'MarketProbAwayNoVig']
    
    # Check if market probs exist, otherwise compute from odds
    if 'MarketProbHomeNoVig' not in df.columns:
        logger.info("Computing market probabilities from odds...")
        # Use closing odds if available, otherwise average
        if 'OddsHome' in df.columns:
            df['_impl_H'] = 1 / df['OddsHome'].fillna(2.0)
            df['_impl_D'] = 1 / df['OddsDraw'].fillna(3.3)
            df['_impl_A'] = 1 / df['OddsAway'].fillna(3.0)
            df['_impl_total'] = df['_impl_H'] + df['_impl_D'] + df['_impl_A']
            df['MarketProbHomeNoVig'] = df['_impl_H'] / df['_impl_total']
            df['MarketProbDrawNoVig'] = df['_impl_D'] / df['_impl_total']
            df['MarketProbAwayNoVig'] = df['_impl_A'] / df['_impl_total']
        else:
            # Fallback: use average odds from B365 or AvgH
            logger.warning("No OddsHome column, using B365 or Avg columns")
            h_odds = df.get('B365H', df.get('AvgH', pd.Series([2.0] * len(df))))
            d_odds = df.get('B365D', df.get('AvgD', pd.Series([3.3] * len(df))))
            a_odds = df.get('B365A', df.get('AvgA', pd.Series([3.0] * len(df))))
            
            impl_h = 1 / h_odds.fillna(2.0)
            impl_d = 1 / d_odds.fillna(3.3)
            impl_a = 1 / a_odds.fillna(3.0)
            impl_total = impl_h + impl_d + impl_a
            
            df['MarketProbHomeNoVig'] = impl_h / impl_total
            df['MarketProbDrawNoVig'] = impl_d / impl_total
            df['MarketProbAwayNoVig'] = impl_a / impl_total
    
    # Fill missing features
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    # Prepare data
    X_features = df[feature_cols].fillna(0.0)
    X_market = df[market_prob_cols].values
    
    # Target: one_hot(result) - market_prob (this is the "residual")
    result_onehot = pd.get_dummies(df['FTR'])[['H', 'D', 'A']].values
    y_residual = result_onehot - X_market
    
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Market probs: {len(market_prob_cols)}")
    logger.info(f"Total input dim: {len(feature_cols) + len(market_prob_cols)}")
    logger.info(f"Samples: {len(df)}")
    
    # Split (time-based)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    X_feat_train = X_features.iloc[:train_end]
    X_feat_val = X_features.iloc[train_end:val_end]
    X_feat_test = X_features.iloc[val_end:]
    
    X_market_train = X_market[:train_end]
    X_market_val = X_market[train_end:val_end]
    X_market_test = X_market[val_end:]
    
    y_train = y_residual[:train_end]
    y_val = y_residual[train_end:val_end]
    y_test = y_residual[val_end:]
    
    # Also keep actual results for evaluation
    y_actual_train = result_onehot[:train_end]
    y_actual_val = result_onehot[train_end:val_end]
    y_actual_test = result_onehot[val_end:]
    
    logger.info(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    # Scale features (not market probs)
    scaler = StandardScaler()
    X_feat_train_scaled = scaler.fit_transform(X_feat_train)
    X_feat_val_scaled = scaler.transform(X_feat_val)
    X_feat_test_scaled = scaler.transform(X_feat_test)
    
    # Combine market probs + scaled features
    X_train = np.hstack([X_market_train, X_feat_train_scaled])
    X_val = np.hstack([X_market_val, X_feat_val_scaled])
    X_test = np.hstack([X_market_test, X_feat_test_scaled])
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Model
    input_dim = X_train.shape[1]  # 3 + 18 = 21
    model = ResidualNN(input_dim=input_dim, hidden_dims=[64, 32, 16], dropout=0.3)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    logger.info("\nTraining residual network...")
    best_val_loss = float('inf')
    best_model_state = None
    patience = 15
    patience_counter = 0
    
    for epoch in range(100):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        # Test predictions
        X_test_tensor = torch.FloatTensor(X_test)
        pred_deltas = model(X_test_tensor).numpy()
        
        # Adjusted probabilities = market + delta
        pred_probs = X_market_test + pred_deltas
        pred_probs = np.clip(pred_probs, 0.01, 0.99)
        pred_probs = pred_probs / pred_probs.sum(axis=1, keepdims=True)
        
        # Accuracy
        pred_class = pred_probs.argmax(axis=1)
        actual_class = y_actual_test.argmax(axis=1)
        accuracy = (pred_class == actual_class).mean()
        
        # Also compute market-only accuracy for comparison
        market_class = X_market_test.argmax(axis=1)
        market_accuracy = (market_class == actual_class).mean()
        
        # Log-loss
        from sklearn.metrics import log_loss, brier_score_loss
        test_logloss = log_loss(actual_class, pred_probs)
        market_logloss = log_loss(actual_class, X_market_test)
        
        # Brier
        test_brier = sum(brier_score_loss(y_actual_test[:, i], pred_probs[:, i]) for i in range(3)) / 3
        market_brier = sum(brier_score_loss(y_actual_test[:, i], X_market_test[:, i]) for i in range(3)) / 3
    
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"\nMarket Baseline:")
    logger.info(f"  Accuracy: {market_accuracy:.4f}")
    logger.info(f"  LogLoss:  {market_logloss:.4f}")
    logger.info(f"  Brier:    {market_brier:.4f}")
    
    logger.info(f"\nResidual Model:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  LogLoss:  {test_logloss:.4f}")
    logger.info(f"  Brier:    {test_brier:.4f}")
    
    logger.info(f"\nImprovement:")
    logger.info(f"  Accuracy: {(accuracy - market_accuracy) * 100:+.2f}%")
    logger.info(f"  LogLoss:  {(test_logloss - market_logloss):+.4f}")
    logger.info(f"  Brier:    {(test_brier - market_brier):+.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = base_dir / 'models' / f'residual_nn_{timestamp}.pt'
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_columns': feature_cols,
        'input_dim': input_dim,
        'hidden_dims': [64, 32, 16],
    }, model_path)
    
    # Create symlink
    latest_path = base_dir / 'models' / 'residual_nn_latest.pt'
    if latest_path.exists():
        latest_path.unlink()
    latest_path.symlink_to(model_path.name)
    
    # Save metadata
    metadata = {
        'model': 'ResidualNN',
        'version': 'v1',
        'train_date': timestamp,
        'input_dim': input_dim,
        'results': {
            'market_accuracy': float(market_accuracy),
            'model_accuracy': float(accuracy),
            'market_logloss': float(market_logloss),
            'model_logloss': float(test_logloss),
            'accuracy_improvement': float(accuracy - market_accuracy),
            'logloss_improvement': float(market_logloss - test_logloss),
        }
    }
    
    meta_path = base_dir / 'models' / f'residual_nn_metadata_{timestamp}.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\n✓ Model saved: {model_path}")
    logger.info(f"✓ Metadata: {meta_path}")
    logger.info("\n" + "=" * 70)
    logger.info("RESIDUAL MODEL TRAINING COMPLETE")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
