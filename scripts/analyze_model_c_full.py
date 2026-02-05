#!/usr/bin/env python3
"""
Model C v2 — Full Analysis Script
==================================
Implements:
1. Isotonic Calibration
2. TimeSeriesCV (5 folds)
3. ROI Backtest
4. Per-league analysis
5. ROI by odds range
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
import json
from datetime import datetime
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.neural_predictor_v2 import ProfessionalNN


class CalibratedTrainer:
    """Trainer with Isotonic Calibration."""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.scaler = StandardScaler()
        self.calibrators = []  # Isotonic calibrators per class
        
    def fit_calibrators(self, X_cal, y_cal):
        """Fit isotonic calibrators on calibration set."""
        logger.info("Fitting isotonic calibrators...")
        
        # Get raw probabilities
        self.model.eval()
        X_tensor = torch.FloatTensor(X_cal).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs_raw = torch.softmax(logits, dim=1).cpu().numpy()
        
        # Fit calibrator for each class
        self.calibrators = []
        for i in range(3):
            y_binary = (y_cal == i).astype(int)
            cal = IsotonicRegression(out_of_bounds='clip')
            cal.fit(probs_raw[:, i], y_binary)
            self.calibrators.append(cal)
            
        logger.info("Calibrators fitted")
    
    def predict_calibrated(self, X):
        """Predict with calibration."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs_raw = torch.softmax(logits, dim=1).cpu().numpy()
        
        if self.calibrators:
            calibrated = np.zeros_like(probs_raw)
            for i, cal in enumerate(self.calibrators):
                calibrated[:, i] = cal.predict(probs_raw[:, i])
            
            # Renormalize
            row_sums = calibrated.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            calibrated = calibrated / row_sums
            return calibrated
        
        return probs_raw


def calculate_roi(predictions, y_true, odds_h, odds_d, odds_a, confidence_threshold=0.0):
    """
    Calculate ROI for 1X2 betting.
    
    Strategy: Bet on the outcome with highest (pred_prob - implied_prob)
    Only bet when confidence >= threshold.
    """
    n = len(predictions)
    
    # Implied probabilities
    implied_h = 1 / odds_h
    implied_d = 1 / odds_d
    implied_a = 1 / odds_a
    total = implied_h + implied_d + implied_a
    implied_h /= total
    implied_d /= total
    implied_a /= total
    
    # Value = predicted - implied
    value_h = predictions[:, 0] - implied_h
    value_d = predictions[:, 1] - implied_d
    value_a = predictions[:, 2] - implied_a
    
    # Confidence = max prediction
    confidence = predictions.max(axis=1)
    
    # Find best value bet per match
    values = np.stack([value_h, value_d, value_a], axis=1)
    best_bet = values.argmax(axis=1)
    best_value = values.max(axis=1)
    
    # Only bet when positive value AND above confidence threshold
    bet_mask = (best_value > 0) & (confidence >= confidence_threshold)
    
    # Calculate returns
    total_stake = 0
    total_return = 0
    wins = 0
    losses = 0
    
    odds_arr = np.stack([odds_h, odds_d, odds_a], axis=1)
    
    for i in range(n):
        if bet_mask[i]:
            bet_type = best_bet[i]
            stake = 1.0  # Flat staking
            total_stake += stake
            
            if y_true[i] == bet_type:
                total_return += stake * odds_arr[i, bet_type]
                wins += 1
            else:
                losses += 1
    
    if total_stake == 0:
        return {
            'roi': 0.0,
            'profit': 0.0,
            'bets': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
        }
    
    profit = total_return - total_stake
    roi = profit / total_stake * 100
    
    return {
        'roi': roi,
        'profit': profit,
        'bets': wins + losses,
        'wins': wins,
        'losses': losses,
        'win_rate': wins / (wins + losses) * 100 if (wins + losses) > 0 else 0,
    }


def run_timeseries_cv(X, y, df, n_splits=5):
    """Run TimeSeriesCV and return aggregated results."""
    logger.info(f"Running TimeSeriesCV with {n_splits} splits...")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    fold = 0
    
    for train_idx, test_idx in tscv.split(X):
        fold += 1
        logger.info(f"Fold {fold}/{n_splits}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        # Handle NaN
        X_train_sc = np.nan_to_num(X_train_sc, 0.0)
        X_test_sc = np.nan_to_num(X_test_sc, 0.0)
        
        # Create and train model
        model = ProfessionalNN(
            input_dim=X_train.shape[1],
            hidden_dims=[128, 64, 64, 32],
            dropout=0.3,
            use_residual=True,
        )
        
        # Quick training for CV (reduced epochs)
        from torch.utils.data import DataLoader, TensorDataset
        import torch.optim as optim
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_sc),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(30):  # Quick training
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            logits = model(torch.FloatTensor(X_test_sc))
            probs = torch.softmax(logits, dim=1).numpy()
        
        y_pred = probs.argmax(axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        
        # ROI
        test_df = df.iloc[test_idx]
        roi_result = calculate_roi(
            probs, y_test,
            test_df['B365H'].values,
            test_df['B365D'].values,
            test_df['B365A'].values,
            confidence_threshold=0.0
        )
        
        results.append({
            'fold': fold,
            'accuracy': accuracy,
            'roi': roi_result['roi'],
            'bets': roi_result['bets'],
            'test_size': len(y_test),
        })
        
        logger.info(f"  Accuracy: {accuracy:.2%}, ROI: {roi_result['roi']:.1f}%, Bets: {roi_result['bets']}")
    
    return results


def analyze_per_league(probs, y_true, df_test):
    """Analyze performance per league."""
    results = {}
    
    for league in df_test['League'].unique():
        mask = df_test['League'] == league
        if mask.sum() < 10:
            continue
            
        y_league = y_true[mask]
        probs_league = probs[mask]
        
        y_pred = probs_league.argmax(axis=1)
        accuracy = accuracy_score(y_league, y_pred)
        
        # ROI
        df_league = df_test[mask]
        roi_result = calculate_roi(
            probs_league, y_league,
            df_league['B365H'].values,
            df_league['B365D'].values,
            df_league['B365A'].values,
        )
        
        results[league] = {
            'matches': int(mask.sum()),
            'accuracy': accuracy,
            'roi': roi_result['roi'],
            'bets': roi_result['bets'],
        }
    
    return results


def analyze_by_odds_range(probs, y_true, df_test):
    """Analyze performance by odds range."""
    # Focus on favorite odds (B365H for home favorites)
    odds_home = df_test['B365H'].values
    
    ranges = [
        ('Heavy Favorite (1.0-1.5)', 1.0, 1.5),
        ('Favorite (1.5-2.0)', 1.5, 2.0),
        ('Slight Favorite (2.0-2.5)', 2.0, 2.5),
        ('Even (2.5-3.5)', 2.5, 3.5),
        ('Underdog (3.5+)', 3.5, 100.0),
    ]
    
    results = {}
    
    for name, low, high in ranges:
        mask = (odds_home >= low) & (odds_home < high)
        if mask.sum() < 10:
            continue
            
        y_range = y_true[mask]
        probs_range = probs[mask]
        
        y_pred = probs_range.argmax(axis=1)
        accuracy = accuracy_score(y_range, y_pred)
        
        # ROI
        df_range = df_test[mask]
        roi_result = calculate_roi(
            probs_range, y_range,
            df_range['B365H'].values,
            df_range['B365D'].values,
            df_range['B365A'].values,
        )
        
        results[name] = {
            'matches': int(mask.sum()),
            'accuracy': accuracy,
            'roi': roi_result['roi'],
            'bets': roi_result['bets'],
        }
    
    return results


def main():
    logger.info("=" * 70)
    logger.info("MODEL C v2 — FULL ANALYSIS")
    logger.info("=" * 70)
    
    base_dir = Path(__file__).parent.parent
    
    # Load data
    df = pd.read_csv(base_dir / 'data' / 'processed' / 'multi_league_features_2021_2024.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Create features
    feature_cols = [
        'HomeEloBefore', 'AwayEloBefore', 'EloDiff', 'EloExpHome', 'EloExpAway',
        'Home_Pts_L5', 'Home_GF_L5', 'Home_GA_L5',
        'Away_Pts_L5', 'Away_GF_L5', 'Away_GA_L5',
        'Home_Overall_Pts_L5', 'Home_Overall_GF_L5', 'Home_Overall_GA_L5',
        'Away_Overall_Pts_L5', 'Away_Overall_GF_L5', 'Away_Overall_GA_L5',
        'Odds_Volatility', 'Market_Consensus', 'Sharp_Divergence',
        'SentimentHome', 'SentimentAway', 'HomeInjury', 'AwayInjury',
    ]
    
    # Add odds-implied
    df['odds_implied_home'] = 1 / df['B365H']
    df['odds_implied_draw'] = 1 / df['B365D']
    df['odds_implied_away'] = 1 / df['B365A']
    total = df['odds_implied_home'] + df['odds_implied_draw'] + df['odds_implied_away']
    df['odds_implied_home'] /= total
    df['odds_implied_draw'] /= total
    df['odds_implied_away'] /= total
    feature_cols.extend(['odds_implied_home', 'odds_implied_draw', 'odds_implied_away'])
    
    # Add value features
    df['value_home'] = df['EloExpHome'] - df['odds_implied_home']
    df['value_away'] = df['EloExpAway'] - df['odds_implied_away']
    feature_cols.extend(['value_home', 'value_away'])
    
    # Filter to available
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].fillna(0)
    y = df['FTR'].map({'H': 0, 'D': 1, 'A': 2}).values
    
    logger.info(f"Dataset: {len(X)} samples, {len(feature_cols)} features")
    
    # Split
    n = len(X)
    train_end = int(n * 0.70)
    cal_end = int(n * 0.80)  # 10% for calibration
    
    X_train, y_train = X.iloc[:train_end], y[:train_end]
    X_cal, y_cal = X.iloc[train_end:cal_end], y[train_end:cal_end]
    X_test, y_test = X.iloc[cal_end:], y[cal_end:]
    df_test = df.iloc[cal_end:]
    
    logger.info(f"Split: Train={len(X_train)}, Cal={len(X_cal)}, Test={len(X_test)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_cal_sc = scaler.transform(X_cal)
    X_test_sc = scaler.transform(X_test)
    
    X_train_sc = np.nan_to_num(X_train_sc, 0.0)
    X_cal_sc = np.nan_to_num(X_cal_sc, 0.0)
    X_test_sc = np.nan_to_num(X_test_sc, 0.0)
    
    # Train model
    logger.info("\n" + "=" * 50)
    logger.info("TRAINING MODEL")
    logger.info("=" * 50)
    
    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim
    
    model = ProfessionalNN(
        input_dim=len(feature_cols),
        hidden_dims=[128, 64, 64, 32],
        dropout=0.3,
        use_residual=True,
    )
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_sc),
        torch.LongTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(50):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    
    # Create calibrated trainer
    trainer = CalibratedTrainer(model)
    trainer.scaler = scaler
    
    # Fit calibrators
    trainer.fit_calibrators(X_cal_sc, y_cal)
    
    # Predict on test
    probs_calibrated = trainer.predict_calibrated(X_test_sc)
    
    # === RESULTS ===
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    
    # 1. Accuracy
    y_pred = probs_calibrated.argmax(axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"\n1. ACCURACY: {accuracy:.2%}")
    
    # 2. Calibration check
    logger.info("\n2. CALIBRATION (after isotonic):")
    for i, name in enumerate(['Home', 'Draw', 'Away']):
        pred_prob = probs_calibrated[:, i].mean()
        actual_freq = (y_test == i).mean()
        gap = pred_prob - actual_freq
        logger.info(f"   {name}: predicted {pred_prob:.1%}, actual {actual_freq:.1%}, gap {gap:+.1%}")
    
    # 3. ROI
    logger.info("\n3. ROI ANALYSIS:")
    for threshold in [0.0, 0.4, 0.5, 0.55, 0.6]:
        roi = calculate_roi(
            probs_calibrated, y_test,
            df_test['B365H'].values,
            df_test['B365D'].values,
            df_test['B365A'].values,
            confidence_threshold=threshold
        )
        logger.info(f"   Threshold {threshold:.0%}: {roi['bets']} bets, ROI={roi['roi']:+.1f}%, Win={roi['win_rate']:.0f}%")
    
    # 4. Per-league
    logger.info("\n4. PER-LEAGUE ANALYSIS:")
    league_results = analyze_per_league(probs_calibrated, y_test, df_test)
    for league, res in sorted(league_results.items(), key=lambda x: x[1]['roi'], reverse=True):
        logger.info(f"   {league}: {res['matches']} matches, Acc={res['accuracy']:.0%}, ROI={res['roi']:+.1f}%")
    
    # 5. By odds range
    logger.info("\n5. BY ODDS RANGE:")
    odds_results = analyze_by_odds_range(probs_calibrated, y_test, df_test)
    for range_name, res in odds_results.items():
        logger.info(f"   {range_name}: {res['matches']} matches, Acc={res['accuracy']:.0%}, ROI={res['roi']:+.1f}%")
    
    # 6. TimeSeriesCV
    logger.info("\n6. TIMESERIES CV (5 folds):")
    cv_results = run_timeseries_cv(X, y, df, n_splits=5)
    
    avg_acc = np.mean([r['accuracy'] for r in cv_results])
    avg_roi = np.mean([r['roi'] for r in cv_results])
    std_acc = np.std([r['accuracy'] for r in cv_results])
    std_roi = np.std([r['roi'] for r in cv_results])
    
    logger.info(f"   Average Accuracy: {avg_acc:.2%} ± {std_acc:.2%}")
    logger.info(f"   Average ROI: {avg_roi:+.1f}% ± {std_roi:.1f}%")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'test_size': len(y_test),
        'accuracy': float(accuracy),
        'calibration': {
            name: {
                'predicted': float(probs_calibrated[:, i].mean()),
                'actual': float((y_test == i).mean())
            }
            for i, name in enumerate(['Home', 'Draw', 'Away'])
        },
        'roi_by_threshold': {
            str(t): calculate_roi(
                probs_calibrated, y_test,
                df_test['B365H'].values, df_test['B365D'].values, df_test['B365A'].values,
                confidence_threshold=t
            ) for t in [0.0, 0.4, 0.5, 0.55, 0.6]
        },
        'per_league': league_results,
        'by_odds_range': odds_results,
        'cv_results': cv_results,
        'cv_summary': {
            'avg_accuracy': float(avg_acc),
            'std_accuracy': float(std_acc),
            'avg_roi': float(avg_roi),
            'std_roi': float(std_roi),
        }
    }
    
    results_file = base_dir / 'models' / 'neural_v2_full_analysis.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    logger.info(f"\nResults saved to {results_file}")
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
