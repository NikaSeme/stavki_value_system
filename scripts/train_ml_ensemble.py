#!/usr/bin/env python3
"""
Simple ensemble training for multi-league models.

Combines:
- CatBoost (Multi-League v6.5)
- Poisson (Multi-League v6.5)

Uses weighted averaging with PlattScaling calibration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.calibration import get_best_calibrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def time_based_split(df, train_frac=0.70, val_frac=0.15):
    """Split data chronologically."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    return train, val, test


from scipy.stats import poisson

class PoissonWrapper:
    """Wrapper for Poisson model loaded from dict."""
    def __init__(self, params):
        self.home_advantage = params['home_advantage']
        self.team_attack = params['team_attack']
        self.team_defense = params['team_defense']
        self.league_avg_goals = params['league_avg_goals']
        
    def predict_match(self, home_team, away_team):
        """Predict match probabilities."""
        # Get team strengths (default to 1.0 if unknown)
        home_attack = self.team_attack.get(home_team, 1.0)
        home_defense = self.team_defense.get(home_team, 1.0)
        away_attack = self.team_attack.get(away_team, 1.0)
        away_defense = self.team_defense.get(away_team, 1.0)
        
        # Expected goals
        lambda_home = self.league_avg_goals * home_attack * away_defense * (1 + self.home_advantage)
        lambda_away = self.league_avg_goals * away_attack * home_defense
        
        # Calculate probabilities
        max_goals = 6
        prob_home = 0.0
        prob_draw = 0.0
        prob_away = 0.0
        
        for i in range(max_goals):
            for j in range(max_goals):
                p = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                if i > j:
                    prob_home += p
                elif i == j:
                    prob_draw += p
                else:
                    prob_away += p
                    
        return [prob_home, prob_draw, prob_away]


class DummyModel:
    """Wrapper for raw probabilities to use with CalibratedClassifierCV."""
    def __init__(self, probs):
        self.probs = probs
        self.classes_ = np.array([0, 1, 2])
        
    def predict_proba(self, X):
        return self.probs

class SimpleEnsemble:
    """
    Weighted average ensemble of CatBoost + Poisson models.
    """
    
    def __init__(self, catboost_weight=0.5):
        self.catboost_weight = catboost_weight
        self.poisson_weight = 1.0 - catboost_weight
        self.catboost = None
        self.poisson = None
        self.calibrator = None
        self.feature_names = None
        
    def load_models(self, catboost_path, poisson_path, metadata_path):
        """Load base models and metadata."""
        logger.info(f"Loading CatBoost from {catboost_path}...")
        self.catboost = joblib.load(catboost_path)
        
        logger.info(f"Loading features from {metadata_path}...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            self.feature_names = metadata.get('features')
            if not self.feature_names:
                raise ValueError("No features found in metadata!")
            logger.info(f"Loaded {len(self.feature_names)} features: {self.feature_names}")
        
        logger.info(f"Loading Poisson from {poisson_path}...")
        with open(poisson_path, 'rb') as f:
            poisson_params = joblib.load(f)
            self.poisson = PoissonWrapper(poisson_params)
            
    def predict_proba(self, df):
        """
        Predict probabilities by averaging base models.
        
        Args:
            df: DataFrame with match data
            
        Returns:
            np.array of shape (n_samples, 3) with probabilities
        """
        # Ensure we only use the features CatBoost expects, in order
        if self.feature_names is None:
            raise ValueError("Features not loaded! Call load_models first.")
            
        # Select and reorder columns matching training data
        cat_features_df = df[self.feature_names].copy()
        
        # Get CatBoost predictions
        cat_probs = self.catboost.predict_proba(cat_features_df)
        
        # Get Poisson predictions
        poisson_probs = []
        for _, row in df.iterrows():
            probs = self.poisson.predict_match(row['HomeTeam'], row['AwayTeam'])
            poisson_probs.append(probs)
        poisson_probs = np.array(poisson_probs)
        
        # Weighted average
        ensemble_probs = (
            self.catboost_weight * cat_probs +
            self.poisson_weight * poisson_probs
        )
        
        # Normalize to sum to 1 (in case of numerical issues)
        ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)
        
        return ensemble_probs
        
    def fit_calibrator(self, df, y_true):
        """Fit Platt scaling calibrator on validation data."""
        logger.info("Fitting calibrator...")
        
        # Get raw ensemble predictions
        probs = self.predict_proba(df)
        
        dummy = DummyModel(probs)
        self.calibrator = get_best_calibrator(dummy)
        self.calibrator.fit(df, y_true)
        
    def predict_calibrated(self, df):
        """Get calibrated predictions."""
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted yet!")
            
        # Get raw predictions
        raw_probs = self.predict_proba(df)
        
        # For PlattScaling, we need to pass through the dummy model
        dummy = DummyModel(raw_probs)
        self.calibrator.base_model = dummy
        
        return self.calibrator.predict_proba(df)


def main():
    logger.info("=" * 70)
    logger.info("SIMPLE ENSEMBLE TRAINING (CatBoost + Poisson)")
    logger.info("=" * 70)
    
    base_dir = Path(__file__).parent.parent
    
    # Load data
    data_file = base_dir / 'data' / 'processed' / 'multi_league_features_2021_2024.csv'
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    logger.info(f"Loaded {len(df)} matches")
    
    # Split
    train_df, val_df, test_df = time_based_split(df)
    
    # Prepare target
    result_map = {'H': 0, 'D': 1, 'A': 2}
    y_val = val_df['FTR'].map(result_map).values
    y_test = test_df['FTR'].map(result_map).values
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Load models
    ensemble = SimpleEnsemble(catboost_weight=0.6)  # 60% CatBoost, 40% Poisson
    
    models_dir = base_dir / 'models'
    catboost_path = models_dir / 'catboost_v1_latest.pkl'
    poisson_path = models_dir / 'poisson_v1_latest.pkl'
    metadata_path = models_dir / 'metadata_v1_latest.json'
    
    ensemble.load_models(catboost_path, poisson_path, metadata_path)
    
    # Fit calibrator on validation set
    ensemble.fit_calibrator(val_df, y_val)
    
    # Evaluate on test set
    logger.info("\\nEvaluating on test set...")
    
    # Raw ensemble
    raw_probs = ensemble.predict_proba(test_df)
    raw_pred = np.argmax(raw_probs, axis=1)
    
    raw_acc = accuracy_score(y_test, raw_pred)
    raw_brier = np.mean([
        brier_score_loss((y_test == i).astype(int), raw_probs[:, i])
        for i in range(3)
    ])
    
    logger.info(f"Raw Ensemble:")
    logger.info(f"  Accuracy: {raw_acc:.2%}")
    logger.info(f"  Brier Score: {raw_brier:.4f}")
    
    # Calibrated ensemble
    cal_probs = ensemble.predict_calibrated(test_df)
    cal_pred = np.argmax(cal_probs, axis=1)
    
    cal_acc = accuracy_score(y_test, cal_pred)
    cal_brier = np.mean([
        brier_score_loss((y_test == i).astype(int), cal_probs[:, i])
        for i in range(3)
    ])
    
    logger.info(f"\\nCalibrated Ensemble:")
    logger.info(f"  Accuracy: {cal_acc:.2%}")
    logger.info(f"  Brier Score: {cal_brier:.4f}")
    
    # Save ensemble
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    ensemble_file = models_dir / f'ensemble_v6_5_{timestamp}.pkl'
    joblib.dump(ensemble, ensemble_file)
    
    # Create symlink
    latest_link = models_dir / 'ensemble_v6_5_latest.pkl'
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(ensemble_file.name)
    
    # Save metadata
    metadata = {
        'version': 'v6.5_multi_league',
        'timestamp': timestamp,
        'components': {
            'catboost': str(catboost_path),
            'poisson': str(poisson_path),
            'catboost_weight': ensemble.catboost_weight,
            'poisson_weight': ensemble.poisson_weight
        },
        'metrics': {
            'test_accuracy': cal_acc,
            'test_brier': cal_brier,
            'raw_accuracy': raw_acc,
            'raw_brier': raw_brier
        },
        'calibration': 'PlattScaling',
        'leagues': df['League'].unique().tolist(),
        'num_matches': len(df)
    }
    
    metadata_file = models_dir / f'ensemble_metadata_v6_5_{timestamp}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\\n✅ Ensemble saved to: {ensemble_file}")
    logger.info(f"✅ Linked to: {latest_link}")
    logger.info("\\n✅ ENSEMBLE TRAINING COMPLETE")
    
    return 0


if __name__ == '__main__':
    exit(main())
