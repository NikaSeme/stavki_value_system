"""
Train Poisson model (Model A) for ensemble.

Calculates team attack/defense strengths from historical data
and generates probability predictions for 3-way outcomes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import pickle
import sys
import argparse
from collections import defaultdict
from scipy.stats import poisson
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_setup import get_logger

logger = get_logger(__name__)


class PoissonMatchPredictor:
    """
    Poisson-based match outcome predictor.
    
    Calculates team strengths (attack/defense) and predicts match
    probabilities using Poisson distributions for goal scoring.
    """
    
    def __init__(self, home_advantage=0.15, time_decay_rate=0.003):
        """
        Initialize Poisson predictor.
        
        Args:
            home_advantage: Home team advantage (default 15%)
            time_decay_rate: Exponential decay rate (default 0.003 = ~50% weight after 231 days)
        """
        self.home_advantage = home_advantage
        self.time_decay_rate = time_decay_rate
        self.team_attack = {}  # Team attack strength
        self.team_defense = {}  # Team defense strength
        self.league_avg_goals = 1.5  # Global league average (fallback)
        self.league_baselines = {}  # Per-league average goals (Task D)
        self.team_leagues = {}  # Map team -> league for predictions
        
    def fit(self, df):
        """
        Calculate team strengths from historical matches with time decay.
        
        Recent matches are weighted exponentially higher using Dixon-Coles approach:
        weight = exp(-decay_rate * days_since_match)
        
        Args:
            df: DataFrame with columns: HomeTeam, AwayTeam, FTHG, FTAG, Date
        """
        logger.info(f"Fitting Poisson model on {len(df)} matches (with time decay)")
        
        # Ensure Date column is datetime with robust error handling
        if 'Date' not in df.columns:
            logger.warning("No Date column found - disabling time decay (using uniform weights)")
            df_copy = df.copy()
            df_copy['Date'] = pd.Timestamp.now()
            df_copy['weight'] = 1.0  # Uniform weights
            self.time_decay_rate = 0.0  # Disable decay
        else:
            df_copy = df.copy()
            
            # Robust date parsing with error coercion
            try:
                df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce')
                
                # Check for and remove invalid dates
                invalid_dates = df_copy['Date'].isna().sum()
                if invalid_dates > 0:
                    logger.warning(f"Found {invalid_dates} rows with invalid dates - dropping them")
                    df_copy = df_copy.dropna(subset=['Date'])
                    
                if len(df_copy) == 0:
                    raise ValueError("No valid dates found in dataset after parsing")
                
                # Calculate days since most recent match
                max_date = df_copy['Date'].max()
                df_copy['days_ago'] = (max_date - df_copy['Date']).dt.days
                
                # Calculate time decay weights
                df_copy['weight'] = np.exp(-self.time_decay_rate * df_copy['days_ago'])
                
                logger.info(f"Time decay: Recent match weight = 1.0, 1-year-old match weight = {np.exp(-self.time_decay_rate * 365):.3f}")
                
            except Exception as e:
                logger.error(f"Date parsing failed: {e} - Falling back to uniform weights")
                df_copy['Date'] = pd.Timestamp.now()
                df_copy['weight'] = 1.0  # Uniform weights as fallback
                self.time_decay_rate = 0.0  # Disable decay

        
        # Calculate weighted average goals
        total_weighted_goals = (df_copy['FTHG'] * df_copy['weight']).sum() + (df_copy['FTAG'] * df_copy['weight']).sum()
        total_weight = df_copy['weight'].sum() * 2  # Each match contributes 2 team-games
        
        # Division by zero protection
        if total_weight > 0:
            self.league_avg_goals = total_weighted_goals / total_weight
        else:
            logger.warning("All weights are zero (extreme decay) - using default league average")
            self.league_avg_goals = 1.5  # Fallback
        
        # Calculate per-league baselines (Task D)
        if 'League' in df_copy.columns:
            logger.info("Calculating per-league baselines...")
            for league in df_copy['League'].unique():
                league_df = df_copy[df_copy['League'] == league]
                league_goals = (league_df['FTHG'] * league_df['weight']).sum() + \
                               (league_df['FTAG'] * league_df['weight']).sum()
                league_weight = league_df['weight'].sum() * 2
                if league_weight > 0:
                    self.league_baselines[league] = league_goals / league_weight
                else:
                    self.league_baselines[league] = self.league_avg_goals
                logger.info(f"  {league}: {self.league_baselines[league]:.3f} goals/match")
                
                # Map teams to leagues
                for team in set(league_df['HomeTeam'].unique()) | set(league_df['AwayTeam'].unique()):
                    self.team_leagues[team] = league
        else:
            logger.warning("No League column found - using global baseline for all predictions")
        
        logger.info(f"Global average goals (time-weighted): {self.league_avg_goals:.3f}")
        
        # Initialize team stats with weighted lists
        team_stats = defaultdict(lambda: {
            'goals_for': [],
            'goals_against': [],
            'weights': []
        })
        
        # Collect team stats with weights
        for _, match in df_copy.iterrows():
            home = match['HomeTeam']
            away = match['AwayTeam']
            home_goals = match['FTHG']
            away_goals = match['FTAG']
            weight = match['weight']
            
            # Home team stats
            team_stats[home]['goals_for'].append(home_goals)
            team_stats[home]['goals_against'].append(away_goals)
            team_stats[home]['weights'].append(weight)
            
            # Away team stats
            team_stats[away]['goals_for'].append(away_goals)
            team_stats[away]['goals_against'].append(home_goals)
            team_stats[away]['weights'].append(weight)
        
        # Calculate weighted attack and defense strengths
        for team, stats in team_stats.items():
            goals_for = np.array(stats['goals_for'])
            goals_against = np.array(stats['goals_against'])
            weights = np.array(stats['weights'])
            
            # Weighted average: sum(x * w) / sum(w)
            if weights.sum() > 0:
                weighted_gf = np.sum(goals_for * weights) / weights.sum()
                weighted_ga = np.sum(goals_against * weights) / weights.sum()
            else:
                weighted_gf = np.mean(goals_for)
                weighted_ga = np.mean(goals_against)
            
            # Attack strength = weighted goals scored / league_avg
            self.team_attack[team] = weighted_gf / self.league_avg_goals
            
            # Defense strength = weighted goals conceded / league_avg
            self.team_defense[team] = weighted_ga / self.league_avg_goals
        
        logger.info(f"Calculated time-weighted strengths for {len(self.team_attack)} teams")
        
    def predict_match(self, home_team, away_team, league=None):
        """
        Predict probabilities for a single match.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            league: Optional league name for per-league baseline
            
        Returns:
            Dict with prob_home, prob_draw, prob_away
        """
        # Get team strengths (default to 1.0 if unknown)
        home_attack = self.team_attack.get(home_team, 1.0)
        home_defense = self.team_defense.get(home_team, 1.0)
        away_attack = self.team_attack.get(away_team, 1.0)
        away_defense = self.team_defense.get(away_team, 1.0)
        
        # Determine league baseline (Task D)
        if league and league in self.league_baselines:
            baseline = self.league_baselines[league]
        elif home_team in self.team_leagues:
            baseline = self.league_baselines.get(self.team_leagues[home_team], self.league_avg_goals)
        else:
            baseline = self.league_avg_goals
        
        # Expected goals
        # lambda_home = baseline * home_attack * away_defense * (1 + home_adv)
        lambda_home = baseline * home_attack * away_defense * (1 + self.home_advantage)
        lambda_away = baseline * away_attack * home_defense
        
        # Calculate probabilities for each scoreline up to 5-5
        max_goals = 6
        prob_home = 0.0
        prob_draw = 0.0
        prob_away = 0.0
        
        for i in range(max_goals):
            for j in range(max_goals):
                # P(home=i, away=j)
                prob_score = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                
                if i > j:
                    prob_home += prob_score
                elif i == j:
                    prob_draw += prob_score
                else:
                    prob_away += prob_score
        
        return {
            'prob_home': prob_home,
            'prob_draw': prob_draw,
            'prob_away': prob_away
        }
    
    def predict(self, df):
        """
        Predict probabilities for all matches in DataFrame.
        
        Args:
            df: DataFrame with HomeTeam, AwayTeam columns (optionally League)
            
        Returns:
            DataFrame with predictions added
        """
        results = []
        
        for _, match in df.iterrows():
            league = match.get('League', None) if hasattr(match, 'get') else None
            probs = self.predict_match(match['HomeTeam'], match['AwayTeam'], league=league)
            results.append(probs)
        
        # Create DataFrame with predictions
        pred_df = pd.DataFrame(results)
        
        return pred_df
    
    def save(self, filepath):
        """Save model to file."""
        model_data = {
            'home_advantage': self.home_advantage,
            'time_decay_rate': self.time_decay_rate,
            'team_attack': self.team_attack,
            'team_defense': self.team_defense,
            'league_avg_goals': self.league_avg_goals,
            'league_baselines': self.league_baselines,  # Task D
            'team_leagues': self.team_leagues,  # Task D
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Poisson model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            home_advantage=model_data['home_advantage'],
            time_decay_rate=model_data.get('time_decay_rate', 0.003)
        )
        model.team_attack = model_data['team_attack']
        model.team_defense = model_data['team_defense']
        model.league_avg_goals = model_data['league_avg_goals']
        # Task D: Restore per-league baselines (backward compatible)
        model.league_baselines = model_data.get('league_baselines', {})
        model.team_leagues = model_data.get('team_leagues', {})
        
        logger.info(f"Poisson model loaded from {filepath}")
        if model.league_baselines:
            logger.info(f"  Per-league baselines: {len(model.league_baselines)} leagues")
        return model


def tune_decay_rate(train_df, val_df, decay_rates=None):
    """
    Grid search to find optimal time_decay_rate (Task G).
    
    Args:
        train_df: Training data
        val_df: Validation data for evaluation
        decay_rates: List of rates to try (default: [0.001, 0.002, 0.003, 0.005, 0.01])
        
    Returns:
        (best_rate, best_brier, results_dict)
    """
    if decay_rates is None:
        decay_rates = [0.001, 0.002, 0.003, 0.005, 0.01]
    
    result_map = {'H': 0, 'D': 1, 'A': 2}
    results = {}
    best_rate, best_brier = None, float('inf')
    
    logger.info(f"Tuning time_decay_rate over {decay_rates}...")
    
    for rate in decay_rates:
        # Train with this rate
        model = PoissonMatchPredictor(home_advantage=0.15, time_decay_rate=rate)
        model.fit(train_df)
        
        # Evaluate on validation set
        preds = model.predict(val_df)
        y_true = val_df['FTR'].map(result_map).values
        
        # Calculate Brier score
        probs_array = preds[['prob_home', 'prob_draw', 'prob_away']].values
        brier_scores = []
        for i in range(3):
            y_binary = (y_true == i).astype(int)
            if len(np.unique(y_binary)) > 1:
                brier_scores.append(brier_score_loss(y_binary, probs_array[:, i]))
        avg_brier = np.mean(brier_scores) if brier_scores else 0.0
        
        results[rate] = avg_brier
        logger.info(f"  rate={rate:.4f}: Brier={avg_brier:.4f}")
        
        if avg_brier < best_brier:
            best_rate, best_brier = rate, avg_brier
    
    logger.info(f"\n✓ Best time_decay_rate: {best_rate:.4f} (Brier={best_brier:.4f})")
    return best_rate, best_brier, results


def main():
    """Train Poisson model and evaluate."""
    # Parse arguments (Task G)
    parser = argparse.ArgumentParser(description='Train Poisson model')
    parser.add_argument('--tune-decay', action='store_true',
                        help='Tune time_decay_rate via grid search')
    parser.add_argument('--decay-rate', type=float, default=0.003,
                        help='Time decay rate (default: 0.003)')
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("POISSON MODEL TRAINING (MODEL A)")
    logger.info("=" * 70)
    
    # Load data
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / 'data' / 'processed' / 'multi_league_features_2021_2024.csv'
    
    logger.info(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    
    # Robust date parsing (matches the safety in fit() method)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    invalid_dates = df['Date'].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"Found {invalid_dates} rows with invalid dates in CSV - dropping them")
        df = df.dropna(subset=['Date'])
    
    df = df.sort_values('Date')
    
    logger.info(f"Total matches: {len(df)}")
    logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Same split as CatBoost for fair comparison
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"\nSplit:")
    logger.info(f"  Train: {len(train_df)} matches ({train_df['Date'].min()} to {train_df['Date'].max()})")
    logger.info(f"  Val:   {len(val_df)} matches ({val_df['Date'].min()} to {val_df['Date'].max()})")
    logger.info(f"  Test:  {len(test_df)} matches ({test_df['Date'].min()} to {test_df['Date'].max()})")
    
    # Task G: Optionally tune time_decay_rate
    if args.tune_decay:
        best_rate, best_brier, tune_results = tune_decay_rate(train_df, val_df)
        decay_rate = best_rate
        logger.info(f"Using tuned decay_rate: {decay_rate}")
    else:
        decay_rate = args.decay_rate
        logger.info(f"Using specified decay_rate: {decay_rate}")
    
    # Train model with time decay
    logger.info("\nTraining Poisson model...")
    model = PoissonMatchPredictor(home_advantage=0.15, time_decay_rate=decay_rate)
    model.fit(train_df)
    
    # Evaluate on all sets
    result_map = {'H': 0, 'D': 1, 'A': 2}
    
    for name, subset in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        logger.info(f"\n{name} Set Evaluation:")
        
        # Predict
        pred_df = model.predict(subset)
        probs = pred_df[['prob_home', 'prob_draw', 'prob_away']].values
        
        # True labels
        y_true = subset['FTR'].map(result_map).values
        
        # Metrics
        y_pred = probs.argmax(axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Brier score (average over classes)
        brier_scores = []
        for i in range(3):
            y_binary = (y_true == i).astype(int)
            brier = brier_score_loss(y_binary, probs[:, i])
            brier_scores.append(brier)
        avg_brier = np.mean(brier_scores)
        
        # Log loss
        logloss = log_loss(y_true, probs)
        
        logger.info(f"  Accuracy: {accuracy:.2%}")
        logger.info(f"  Brier Score: {avg_brier:.4f}")
        logger.info(f"  Log Loss: {logloss:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = base_dir / 'models'
    model_file = model_dir / f'poisson_v1_{timestamp}.pkl'
    model_latest = model_dir / 'poisson_v1_latest.pkl'
    
    model.save(model_file)
    
    # Create symlink for latest
    if model_latest.exists():
        model_latest.unlink()
    model_latest.symlink_to(model_file.name)
    
    # Save test predictions for ensemble training
    test_pred = model.predict(test_df)
    test_pred['Date'] = test_df['Date'].values
    test_pred['HomeTeam'] = test_df['HomeTeam'].values
    test_pred['AwayTeam'] = test_df['AwayTeam'].values
    test_pred['FTR'] = test_df['FTR'].values
    
    pred_file = model_dir / f'poisson_test_predictions_{timestamp}.csv'
    test_pred.to_csv(pred_file, index=False)
    logger.info(f"Test predictions saved to {pred_file}")
    
    # Save metadata
    metadata = {
        'model': 'Poisson',
        'version': 'v1',
        'train_date': timestamp,
        'home_advantage': model.home_advantage,
        'league_avg_goals': model.league_avg_goals,
        'num_teams': len(model.team_attack),
        'train_matches': len(train_df),
        'val_matches': len(val_df),
        'test_matches': len(test_df),
        'metrics': {
            'test': {
                'accuracy': float(accuracy),
                'brier_score': float(avg_brier),
                'log_loss': float(logloss)
            }
        }
    }
    
    meta_file = model_dir / f'poisson_metadata_{timestamp}.json'
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\n{'=' * 70}")
    logger.info("✅ POISSON MODEL TRAINING COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"Model: {model_file}")
    logger.info(f"Predictions: {pred_file}")
    logger.info(f"Metadata: {meta_file}")


if __name__ == '__main__':
    main()
