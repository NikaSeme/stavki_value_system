"""
Meta-Filter: Bet Quality Classifier

A second-stage model that predicts whether a "value bet" signal
will actually be profitable. This filters out "fake value" where
EV looks positive but historically loses.

Input features:
- ev_pct: Expected value percentage
- odds: Decimal odds
- model_confidence: Max probability from model
- divergence: model_prob - market_prob
- volatility: Odds volatility
- league_id: Categorical (encoded)
- time_to_kickoff: Hours until match

Target: Binary — did the bet profit? (1 = yes, 0 = no)

Model: LightGBM (handles categorical, less overfit than NN)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)
import pickle

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import GradientBoostingClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetaFilter:
    """
    Binary classifier to filter profitable bets.
    
    Usage:
        filter = MetaFilter()
        filter.load()
        
        # For a potential bet:
        should_bet = filter.should_bet(
            ev_pct=0.08,
            odds=2.5,
            model_confidence=0.42,
            divergence=0.05,
            volatility=0.02,
            league='soccer_epl',
            time_to_kickoff=12.0
        )
    """
    
    # League encoding
    LEAGUE_MAP = {
        'soccer_epl': 0,
        'soccer_germany_bundesliga': 1,
        'soccer_spain_la_liga': 2,
        'soccer_italy_serie_a': 3,
        'soccer_france_ligue_one': 4,
        'soccer_efl_champ': 5,
        'E0': 0, 'D1': 1, 'SP1': 2, 'I1': 3, 'F1': 4, 'ENG2': 5,
    }
    
    def __init__(self, model_path: str = 'models/meta_filter_latest.pkl'):
        self.model_path = Path(model_path)
        self.model = None
        self.threshold = 0.5  # Probability threshold for betting
        self.feature_names = [
            'ev_pct', 'odds', 'model_confidence', 'divergence',
            'volatility', 'league_id', 'time_to_kickoff', 'odds_bucket'
        ]
    
    def load(self):
        """Load trained model."""
        if not self.model_path.exists():
            logger.warning(f"Meta-filter not found: {self.model_path}")
            return False
        
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.threshold = data.get('threshold', 0.5)
        self.feature_names = data.get('feature_names', self.feature_names)
        
        logger.info(f"✓ Meta-filter loaded (threshold={self.threshold:.2f})")
        return True
    
    def _odds_bucket(self, odds: float) -> int:
        """Categorize odds into buckets."""
        if odds < 1.5:
            return 0
        elif odds < 2.0:
            return 1
        elif odds < 3.0:
            return 2
        elif odds < 5.0:
            return 3
        else:
            return 4
    
    def _encode_league(self, league: str) -> int:
        """Encode league to numeric."""
        return self.LEAGUE_MAP.get(league, 5)
    
    def predict_proba(
        self,
        ev_pct: float,
        odds: float,
        model_confidence: float,
        divergence: float,
        volatility: float = 0.0,
        league: str = 'soccer_epl',
        time_to_kickoff: float = 24.0
    ) -> float:
        """
        Predict probability that bet will be profitable.
        
        Returns:
            Probability (0-1) that bet is profitable
        """
        if self.model is None:
            # No filter loaded — accept all bets
            return 1.0
        
        X = np.array([[
            ev_pct,
            odds,
            model_confidence,
            divergence,
            volatility,
            self._encode_league(league),
            time_to_kickoff,
            self._odds_bucket(odds)
        ]])
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[0, 1]
        else:
            return self.model.predict(X)[0]
    
    def should_bet(self, **kwargs) -> bool:
        """
        Decide if bet should be placed.
        
        Returns:
            True if bet passes filter
        """
        prob = self.predict_proba(**kwargs)
        return prob >= self.threshold


def train_meta_filter(
    historical_bets: pd.DataFrame,
    output_path: str = 'models/meta_filter_latest.pkl'
):
    """
    Train meta-filter on historical betting data.
    
    Args:
        historical_bets: DataFrame with columns:
            - ev_pct: Expected value at time of bet
            - odds: Decimal odds
            - model_confidence: Max model probability
            - divergence: model_prob - market_prob
            - volatility: Odds volatility
            - league: League code
            - time_to_kickoff: Hours
            - profit: Actual profit (positive = win, negative = loss)
            
    Returns:
        Trained MetaFilter
    """
    logger.info("Training meta-filter...")
    
    # Prepare features
    df = historical_bets.copy()
    
    # Create target: 1 if profitable, 0 otherwise
    df['target'] = (df['profit'] > 0).astype(int)
    
    # Encode league
    filter_obj = MetaFilter()
    df['league_id'] = df['league'].apply(filter_obj._encode_league)
    
    # Odds bucket
    df['odds_bucket'] = df['odds'].apply(filter_obj._odds_bucket)
    
    # Features
    feature_cols = [
        'ev_pct', 'odds', 'model_confidence', 'divergence',
        'volatility', 'league_id', 'time_to_kickoff', 'odds_bucket'
    ]
    
    # Fill missing
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    X = df[feature_cols].fillna(0.0)
    y = df['target']
    
    logger.info(f"Training samples: {len(X)}")
    logger.info(f"Win rate: {y.mean():.2%}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train
    if HAS_LIGHTGBM:
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_child_samples=20,
            random_state=42,
            verbose=-1
        )
    else:
        logger.warning("LightGBM not available, using sklearn GradientBoosting")
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    logger.info("\nMeta-Filter Evaluation:")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1:        {f1:.4f}")
    logger.info(f"  AUC:       {auc:.4f}")
    
    # Optimize threshold for profitability
    best_threshold = 0.5
    best_roi = -999
    
    for thresh in np.arange(0.3, 0.8, 0.05):
        predictions = (y_prob >= thresh).astype(int)
        if predictions.sum() == 0:
            continue
        
        # Simulated ROI: (wins - losses) / bets
        wins = (predictions & y_test.values).sum()
        total_bets = predictions.sum()
        roi = (wins / total_bets) - 0.5  # Assuming 50% base rate
        
        if roi > best_roi:
            best_roi = roi
            best_threshold = thresh
    
    logger.info(f"\nOptimal threshold: {best_threshold:.2f}")
    
    # Save
    output_path = Path(output_path)
    with open(output_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'threshold': best_threshold,
            'feature_names': feature_cols,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }
        }, f)
    
    logger.info(f"✓ Meta-filter saved: {output_path}")
    
    return filter_obj


def generate_synthetic_training_data():
    """
    Generate synthetic historical bet data for training.
    
    In production, this would come from actual bet history.
    This is a placeholder for development/testing.
    """
    np.random.seed(42)
    n_samples = 5000
    
    # Generate features
    ev_pct = np.random.uniform(0.02, 0.30, n_samples)
    odds = np.random.uniform(1.3, 10.0, n_samples)
    model_confidence = np.random.uniform(0.25, 0.70, n_samples)
    divergence = np.random.uniform(-0.10, 0.20, n_samples)
    volatility = np.random.uniform(0.0, 0.10, n_samples)
    time_to_kickoff = np.random.uniform(1, 48, n_samples)
    
    leagues = ['E0', 'D1', 'SP1', 'I1', 'F1', 'ENG2']
    league = np.random.choice(leagues, n_samples)
    
    # Generate profit with realistic patterns
    # High EV + high confidence + low volatility = more likely to win
    base_win_prob = 0.35 + (ev_pct * 0.5) + (model_confidence * 0.2) - (volatility * 2)
    base_win_prob = np.clip(base_win_prob, 0.2, 0.7)
    
    # But high odds are harder to hit
    odds_penalty = (odds - 2.0) * 0.05
    win_prob = base_win_prob - odds_penalty
    win_prob = np.clip(win_prob, 0.15, 0.65)
    
    # Generate outcomes
    won = np.random.random(n_samples) < win_prob
    profit = np.where(won, odds - 1, -1.0)
    
    df = pd.DataFrame({
        'ev_pct': ev_pct,
        'odds': odds,
        'model_confidence': model_confidence,
        'divergence': divergence,
        'volatility': volatility,
        'league': league,
        'time_to_kickoff': time_to_kickoff,
        'profit': profit
    })
    
    return df


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    logger.info("=" * 60)
    logger.info("META-FILTER TRAINING")
    logger.info("=" * 60)
    
    # Check for real historical data
    base_dir = Path(__file__).parent.parent
    history_file = base_dir / 'data' / 'bet_history.csv'
    
    if history_file.exists():
        logger.info(f"Loading historical bets from {history_file}")
        df = pd.read_csv(history_file)
    else:
        logger.info("No historical data found, using synthetic data for development")
        df = generate_synthetic_training_data()
        logger.info(f"Generated {len(df)} synthetic samples")
    
    # Train
    train_meta_filter(df, str(base_dir / 'models' / 'meta_filter_latest.pkl'))
    
    # Test load
    logger.info("\nTesting load...")
    mf = MetaFilter(str(base_dir / 'models' / 'meta_filter_latest.pkl'))
    mf.load()
    
    # Test prediction
    prob = mf.predict_proba(
        ev_pct=0.10,
        odds=2.5,
        model_confidence=0.45,
        divergence=0.08,
        volatility=0.02,
        league='soccer_epl',
        time_to_kickoff=12
    )
    logger.info(f"Test prediction: {prob:.3f} (threshold: {mf.threshold:.2f})")
    logger.info(f"Should bet: {mf.should_bet(ev_pct=0.10, odds=2.5, model_confidence=0.45, divergence=0.08)}")
    
    logger.info("\n✓ Meta-filter ready")
