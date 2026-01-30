"""
Poisson model for football match outcome prediction.

Uses Poisson distribution to estimate expected goals and calculate
Home/Draw/Away probabilities based on historical performance.
Now integrates dynamic Elo ratings and per-league Home Advantage.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import yaml

import numpy as np
import pandas as pd
from scipy.stats import poisson

from ..logging_setup import get_logger

logger = get_logger(__name__)


# Constants
DEFAULT_HOME_ADVANTAGE = 1.15  # Fallback if specific league config missing
LEAGUE_AVG_GOALS = 1.5  # Typical goals per team per match
MAX_GOALS = 10  # Maximum goals to consider in probability calculations
BASE_ELO = 1500.0
ELO_DIVISOR = 1000.0  # Sensitivity of Elo impact on Lambda


def load_league_config() -> Dict[str, float]:
    """Load per-league home advantage from config file."""
    try:
        config_path = Path("config/leagues.yaml")
        if not config_path.exists():
            # Try looking relative to this file if not found
            config_path = Path(__file__).parent.parent.parent / "config" / "leagues.yaml"
            
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                
            ha_map = {}
            if 'soccer' in config:
                for league in config['soccer']:
                    key = league.get('key')
                    ha = league.get('home_advantage', DEFAULT_HOME_ADVANTAGE)
                    if key:
                        ha_map[key] = float(ha)
            return ha_map
    except Exception as e:
        logger.warning(f"Failed to load league config: {e}")
    
    return {}


def estimate_lambda(
    goals_for_avg: float,
    goals_against_avg: float,
    opponent_goals_for_avg: float,
    opponent_goals_against_avg: float,
    is_home: bool = True,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
    elo_rating: Optional[float] = None
) -> float:
    """
    Estimate expected goals (λ) using team stats, opponent stats, and Elo.
    
    Args:
        goals_for_avg: Team's average goals scored
        goals_against_avg: Team's average goals conceded
        opponent_goals_for_avg: Opponent's average goals scored  
        opponent_goals_against_avg: Opponent's average goals conceded
        is_home: Whether team is playing at home
        home_advantage: Specific home advantage multiplier
        elo_rating: Current Elo rating of the team (optional)
        
    Returns:
        Expected goals (lambda parameter)
    """
    # Handle missing data (teams with no history)
    if np.isnan(goals_for_avg) or np.isnan(opponent_goals_against_avg):
        lambda_param = LEAGUE_AVG_GOALS
        # If we have Elo but no goal history, we can still adjust slightly
        if elo_rating and not np.isnan(elo_rating):
             elo_multiplier = 1.0 + (elo_rating - BASE_ELO) / ELO_DIVISOR
             lambda_param *= elo_multiplier
    else:
        # Team's attack strength vs opponent's defense strength
        # Normalized by league average
        attack_strength = goals_for_avg / LEAGUE_AVG_GOALS if LEAGUE_AVG_GOALS > 0 else 1.0
        defense_weakness = opponent_goals_against_avg / LEAGUE_AVG_GOALS if LEAGUE_AVG_GOALS > 0 else 1.0
        
        lambda_param = LEAGUE_AVG_GOALS * attack_strength * defense_weakness
        
        # Apply Elo adjustment if available
        # Elo acts as a "dynamic form" modifier to the long-term averages
        if elo_rating and not np.isnan(elo_rating):
            # Example: Elo 1600 -> +0.1 multiplier (10% boost)
            # Example: Elo 1400 -> -0.1 multiplier (10% penalty)
            elo_multiplier = 1.0 + (elo_rating - BASE_ELO) / ELO_DIVISOR
            # Clip multiplier to reasonable range (0.5 to 1.5) to prevent extreme outliers
            elo_multiplier = max(0.5, min(1.5, elo_multiplier))
            lambda_param *= elo_multiplier
    
    # Apply home advantage if applicable
    if is_home:
        lambda_param *= home_advantage
    
    # Ensure positive value
    return max(lambda_param, 0.1)


def calculate_match_probabilities(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = MAX_GOALS
) -> Dict[str, float]:
    """
    Calculate Home/Draw/Away probabilities using Poisson distribution.
    
    For each possible score (i, j), calculates P(home=i) * P(away=j),
    then sums probabilities for each outcome.
    
    Args:
        lambda_home: Expected goals for home team
        lambda_away: Expected goals for away team
        max_goals: Maximum goals to consider (default: 10)
        
    Returns:
        Dictionary with prob_home, prob_draw, prob_away
    """
    prob_home_win = 0.0
    prob_draw = 0.0
    prob_away_win = 0.0
    
    # Calculate probabilities for all score combinations
    for i in range(max_goals + 1):
        prob_home_i = poisson.pmf(i, lambda_home)
        
        for j in range(max_goals + 1):
            prob_away_j = poisson.pmf(j, lambda_away)
            
            # Joint probability of this exact score
            prob_score = prob_home_i * prob_away_j
            
            # Classify outcome
            if i > j:
                prob_home_win += prob_score
            elif i == j:
                prob_draw += prob_score
            else:
                prob_away_win += prob_score
    
    # Normalize to ensure sum = 1.0 (handle floating point errors)
    total = prob_home_win + prob_draw + prob_away_win
    
    if total > 0:
        prob_home_win /= total
        prob_draw /= total
        prob_away_win /= total
    else:
        # Fallback to equal probabilities
        prob_home_win = prob_draw = prob_away_win = 1/3
    
    return {
        'prob_home': prob_home_win,
        'prob_draw': prob_draw,
        'prob_away': prob_away_win,
    }


def predict_match(
    match_features: pd.Series, 
    ha_map: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Predict match outcome probabilities for a single match.
    
    Args:
        match_features: Series with home/away team statistics
        ha_map: Dictionary mapping league keys to home advantage values
        
    Returns:
        Dictionary with lambda_home, lambda_away, prob_home, prob_draw, prob_away
    """
    # Extract features
    home_goals_for = match_features.get('home_goals_for_avg_5', np.nan)
    home_goals_against = match_features.get('home_goals_against_avg_5', np.nan)
    away_goals_for = match_features.get('away_goals_for_avg_5', np.nan)
    away_goals_against = match_features.get('away_goals_against_avg_5', np.nan)
    
    # Extract Elo
    home_elo = match_features.get('HomeEloBefore', np.nan)
    away_elo = match_features.get('AwayEloBefore', np.nan)
    
    # Determine Home Advantage
    league_key = match_features.get('League', None)
    # Handle case where League might be 'Unknown' or missing
    if not league_key or league_key == 'Unknown':
        # Try to infer from filename/metadata if possible, otherwise default
        current_ha = DEFAULT_HOME_ADVANTAGE
    else:
        current_ha = ha_map.get(league_key, DEFAULT_HOME_ADVANTAGE) if ha_map else DEFAULT_HOME_ADVANTAGE

    # Estimate expected goals
    lambda_home = estimate_lambda(
        goals_for_avg=home_goals_for,
        goals_against_avg=home_goals_against,
        opponent_goals_for_avg=away_goals_for,
        opponent_goals_against_avg=away_goals_against,
        is_home=True,
        home_advantage=current_ha,
        elo_rating=home_elo
    )
    
    lambda_away = estimate_lambda(
        goals_for_avg=away_goals_for,
        goals_against_avg=away_goals_against,
        opponent_goals_for_avg=home_goals_for,
        opponent_goals_against_avg=home_goals_against,
        is_home=False,
        home_advantage=current_ha,  # Not used for away, but passed for consistency in signature
        elo_rating=away_elo
    )
    
    # Calculate probabilities
    probs = calculate_match_probabilities(lambda_home, lambda_away)
    
    # Add lambda values and metadata to output
    result = {
        'lambda_home': lambda_home,
        'lambda_away': lambda_away,
        'used_home_advantage': current_ha,
        **probs
    }
    
    return result


def predict_dataset(
    features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate Poisson predictions for all matches in dataset.
    
    Args:
        features_df: DataFrame with match features
        
    Returns:
        DataFrame with original features plus predictions
    """
    logger.info(f"Generating Poisson predictions for {len(features_df)} matches")
    
    # Load league config
    ha_map = load_league_config()
    logger.info(f"Loaded Home Advantage config for {len(ha_map)} leagues")
    
    predictions = []
    
    for idx, match in features_df.iterrows():
        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(features_df)} matches")
        
        pred = predict_match(match, ha_map)
        predictions.append(pred)
    
    # Add predictions to dataframe
    predictions_df = features_df.copy()
    predictions_df['lambda_home'] = [p['lambda_home'] for p in predictions]
    predictions_df['lambda_away'] = [p['lambda_away'] for p in predictions]
    predictions_df['prob_home'] = [p['prob_home'] for p in predictions]
    predictions_df['prob_draw'] = [p['prob_draw'] for p in predictions]
    predictions_df['prob_away'] = [p['prob_away'] for p in predictions]
    
    # Verify probabilities sum to 1.0
    prob_sums = (
        predictions_df['prob_home'] + 
        predictions_df['prob_draw'] + 
        predictions_df['prob_away']
    )
    
    max_deviation = abs(prob_sums - 1.0).max()
    logger.info(f"Maximum probability sum deviation: {max_deviation:.6f}")
    
    if max_deviation > 0.001:
        logger.warning(f"Some probabilities deviate from 1.0 by more than 0.001")
    
    logger.info("Poisson predictions complete")
    
    return predictions_df


class PoissonModel:
    """
    Poisson model for match outcome prediction.
    
    Wrapper class for extensibility.
    Now automatically loads league configuration for dynamic Home Advantage.
    """
    
    def __init__(
        self,
        home_advantage: float = DEFAULT_HOME_ADVANTAGE,
        league_avg_goals: float = LEAGUE_AVG_GOALS,
        max_goals: int = MAX_GOALS
    ):
        """
        Initialize Poisson model.
        
        Args:
            home_advantage: Default Home team advantage (used if league specific not found)
            league_avg_goals: League average goals per team
            max_goals: Maximum goals to consider
        """
        self.home_advantage = home_advantage
        self.league_avg_goals = league_avg_goals
        self.max_goals = max_goals
        
        logger.info(
            f"Initialized Poisson model: "
            f"default_home_advantage={home_advantage:.2f}, "
            f"league_avg={league_avg_goals:.2f}, "
            f"max_goals={max_goals}"
        )
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for dataset.
        
        Args:
            features_df: DataFrame with match features
            
        Returns:
            DataFrame with predictions
        """
        return predict_dataset(features_df)
    
    def predict_from_file(
        self,
        input_file: Path,
        output_file: Path
    ) -> Dict[str, any]:
        """
        Load features, predict, and save results.
        
        Args:
            input_file: Path to features CSV
            output_file: Path to save predictions CSV
            
        Returns:
            Statistics dictionary
        """
        logger.info(f"Loading features from {input_file}")
        features_df = pd.read_csv(input_file)
        
        logger.info(f"Loaded {len(features_df)} matches")
        
        # Generate predictions
        predictions_df = self.predict(features_df)
        
        # Save to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(output_file, index=False)
        
        logger.info(f"Saved predictions to {output_file}")
        
        # Calculate statistics
        stats = {
            'total_matches': len(predictions_df),
            'avg_prob_home': predictions_df['prob_home'].mean(),
            'avg_prob_draw': predictions_df['prob_draw'].mean(),
            'avg_prob_away': predictions_df['prob_away'].mean(),
            'avg_lambda_home': predictions_df['lambda_home'].mean(),
            'avg_lambda_away': predictions_df['lambda_away'].mean(),
        }
        
        return stats
