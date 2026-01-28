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
from collections import defaultdict
from scipy.stats import poisson
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.league_avg_goals = 1.5  # League average
        
    def fit(self, df):
        """
        Calculate team strengths from historical matches with time decay.
        
        Recent matches are weighted exponentially higher using Dixon-Coles approach:
        weight = exp(-decay_rate * days_since_match)
        
        Args:
            df: DataFrame with columns: HomeTeam, AwayTeam, FTHG, FTAG, Date
        """
        logger.info(f"Fitting Poisson model on {len(df)} matches (with time decay)")
        
        # Ensure Date column is datetime
        if 'Date' not in df.columns:
            logger.warning("No Date column found - using simple averages without decay")
            df_copy = df.copy()
            df_copy['Date'] = pd.Timestamp.now()
        else:
            df_copy = df.copy()
            df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        
        # Calculate days since most recent match
        max_date = df_copy['Date'].max()
        df_copy['days_ago'] = (max_date - df_copy['Date']).dt.days
        
        # Calculate time decay weights
        df_copy['weight'] = np.exp(-self.time_decay_rate * df_copy['days_ago'])
        
        logger.info(f"Time decay: Recent match weight = 1.0, 1-year-old match weight = {np.exp(-self.time_decay_rate * 365):.3f}")
        
        # Calculate weighted average goals
        total_weighted_goals = (df_copy['FTHG'] * df_copy['weight']).sum() + (df_copy['FTAG'] * df_copy['weight']).sum()
        total_weight = df_copy['weight'].sum() * 2  # Each match contributes 2 team-games
        self.league_avg_goals = total_weighted_goals / total_weight
        
        logger.info(f"League average goals (time-weighted): {self.league_avg_goals:.3f}")
        
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
        
    def predict_match(self, home_team, away_team):
        """
        Predict probabilities for a single match.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Dict with prob_home, prob_draw, prob_away
        """
        # Get team strengths (default to 1.0 if unknown)
        home_attack = self.team_attack.get(home_team, 1.0)
        home_defense = self.team_defense.get(home_team, 1.0)
        away_attack = self.team_attack.get(away_team, 1.0)
        away_defense = self.team_defense.get(away_team, 1.0)
        
        # Expected goals
        # lambda_home = league_avg * home_attack * away_defense * (1 + home_adv)
        lambda_home = self.league_avg_goals * home_attack * away_defense * (1 + self.home_advantage)
        lambda_away = self.league_avg_goals * away_attack * home_defense
        
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
            df: DataFrame with HomeTeam, AwayTeam columns
            
        Returns:
            DataFrame with predictions added
        """
        results = []
        
        for _, match in df.iterrows():
            probs = self.predict_match(match['HomeTeam'], match['AwayTeam'])
            results.append(probs)
        
        # Create DataFrame with predictions
        pred_df = pd.DataFrame(results)
        
        return pred_df
    
    def save(self, filepath):
        """Save model to file."""
        model_data = {
            'home_advantage': self.home_advantage,
            'team_attack': self.team_attack,
            'team_defense': self.team_defense,
            'league_avg_goals': self.league_avg_goals
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
        
        model = cls(home_advantage=model_data['home_advantage'])
        model.team_attack = model_data['team_attack']
        model.team_defense = model_data['team_defense']
        model.league_avg_goals = model_data['league_avg_goals']
        
        logger.info(f"Poisson model loaded from {filepath}")
        return model


def main():
    """Train Poisson model and evaluate."""
    logger.info("=" * 70)
    logger.info("POISSON MODEL TRAINING (MODEL A)")
    logger.info("=" * 70)
    
    # Load data
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / 'data' / 'processed' / 'multi_league_features_2021_2024.csv'
    
    logger.info(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
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
    
    # Train model
    logger.info("\nTraining Poisson model...")
    model = PoissonMatchPredictor(home_advantage=0.15)
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
