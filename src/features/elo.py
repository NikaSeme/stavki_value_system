"""
Elo Rating System for Football Teams

Tracks team strength over time using Elo ratings.
Updates ratings after each match based on result and expected outcome.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EloRating:
    """
    Elo rating system for tracking team strength.
    
    Attributes:
        ratings: Current Elo rating for each team
        k_factor: Rating change sensitivity (higher = more volatile)
        home_advantage: Elo points added to home team
        initial_rating: Starting rating for new teams
    """
    
    def __init__(
        self,
        k_factor: float = 20,
        home_advantage: float = 100,
        initial_rating: float = 1500
    ):
        self.ratings: Dict[str, float] = {}
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        
        # History for tracking
        self.history: list = []
    
    def get_rating(self, team: str) -> float:
        """Get current Elo rating for a team."""
        return self.ratings.get(team, self.initial_rating)
    
    def expected_score(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """
        Calculate expected match score (probability of win).
        
        Returns:
            (expected_home, expected_away) each in [0, 1]
        """
        home_elo = self.get_rating(home_team) + self.home_advantage
        away_elo = self.get_rating(away_team)
        
        # Elo formula: E = 1 / (1 + 10^((R_opponent - R_team) / 400))
        exp_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        exp_away = 1 - exp_home
        
        return exp_home, exp_away
    
    def update(
        self,
        home_team: str,
        away_team: str,
        result: str,
        date: Optional[pd.Timestamp] = None
    ) -> Tuple[float, float]:
        """
        Update Elo ratings after a match.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            result: Match result ('H' for home win, 'D' for draw, 'A' for away win)
            date: Match date (for history tracking)
            
        Returns:
            (new_home_rating, new_away_rating)
        """
        # Get current ratings
        home_elo_before = self.get_rating(home_team)
        away_elo_before = self.get_rating(away_team)
        
        # Calculate expected scores
        exp_home, exp_away = self.expected_score(home_team, away_team)
        
        # Actual scores based on result
        score_map = {
            'H': (1.0, 0.0),  # Home win
            'D': (0.5, 0.5),  # Draw
            'A': (0.0, 1.0),  # Away win
        }
        
        if result not in score_map:
            raise ValueError(f"Invalid result: {result}. Must be 'H', 'D', or 'A'")
        
        act_home, act_away = score_map[result]
        
        # Update ratings
        home_elo_after = home_elo_before + self.k_factor * (act_home - exp_home)
        away_elo_after = away_elo_before + self.k_factor * (act_away - exp_away)
        
        self.ratings[home_team] = home_elo_after
        self.ratings[away_team] = away_elo_after
       
        # Record history
        self.history.append({
            'date': date,
            'home_team': home_team,
            'away_team': away_team,
            'result': result,
            'home_elo_before': home_elo_before,
            'away_elo_before': away_elo_before,
            'home_elo_after': home_elo_after,
            'away_elo_after': away_elo_after,
            'exp_home': exp_home,
            'exp_away': exp_away,
        })
        
        return home_elo_after, away_elo_after
    
    def get_ratings_df(self) -> pd.DataFrame:
        """Get current ratings as DataFrame."""
        return pd.DataFrame([
            {'team': team, 'elo': rating}
            for team, rating in sorted(self.ratings.items(), key=lambda x: -x[1])
        ])
    
    def get_history_df(self) -> pd.DataFrame:
        """Get rating history as DataFrame."""
        return pd.DataFrame(self.history)


def calculate_elo_for_dataset(
    df: pd.DataFrame,
    k_factor: float = 20,
    home_advantage: float = 100
) -> pd.DataFrame:
    """
    Calculate Elo ratings for all matches in a dataset.
    
    Args:
        df: DataFrame with columns: Date, HomeTeam, AwayTeam, FTR
        k_factor: Elo K-factor
        home_advantage: Home advantage in Elo points
        
    Returns:
        DataFrame with added columns: HomeElo, AwayElo, HomeEloBefore, AwayEloBefore
    """
    # Initialize Elo system
    elo = EloRating(k_factor=k_factor, home_advantage=home_advantage)
    
    # Sort by date
    df = df.sort_values('Date').copy()
    
    # Track ratings before each match
    home_elo_before = []
    away_elo_before = []
    home_elo_after = []
    away_elo_after = []
    exp_home = []
    exp_away = []
    
    logger.info(f"Calculating Elo ratings for {len(df)} matches...")
    
    for idx, row in df.iterrows():
        # Get ratings before match
        home_rating = elo.get_rating(row['HomeTeam'])
        away_rating = elo.get_rating(row['AwayTeam'])
        
        # Expected scores
        e_home, e_away = elo.expected_score(row['HomeTeam'], row['AwayTeam'])
        
        # Update ratings
        home_after, away_after = elo.update(
            row['HomeTeam'],
            row['AwayTeam'],
            row['FTR'],
            row['Date']
        )
        
        # Store
        home_elo_before.append(home_rating)
        away_elo_before.append(away_rating)
        home_elo_after.append(home_after)
        away_elo_after.append(away_after)
        exp_home.append(e_home)
        exp_away.append(e_away)
    
    # Add to DataFrame
    df['HomeEloBefore'] = home_elo_before
    df['AwayEloBefore'] = away_elo_before
    df['HomeEloAfter'] = home_elo_after
    df['AwayEloAfter'] = away_elo_after
    df['EloExpHome'] = exp_home
    df['EloExpAway'] = exp_away
    df['EloDiff'] = df['HomeEloBefore'] - df['AwayEloBefore']
    
    logger.info(f"✓ Elo calculation complete")
    logger.info(f"  Elo range: {df['HomeEloBefore'].min():.0f} - {df['HomeEloBefore'].max():.0f}")
    logger.info(f"  Teams tracked: {len(elo.ratings)}")
    
    return df


def main():
    """Test Elo calculation on historical data."""
    from pathlib import Path
    
    # Load processed data
    base_dir = Path(__file__).parent.parent.parent  # Go up to project root
    data_file = base_dir / 'data' / 'processed' / 'epl_historical_2021_2024.csv'
    
    logger.info(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate Elo
    df_with_elo = calculate_elo_for_dataset(df, k_factor=20, home_advantage=100)
    
    # Save
    output_file = data_file.parent / 'epl_with_elo_2021_2024.csv'
    df_with_elo.to_csv(output_file, index=False)
    logger.info(f"✓ Saved to {output_file}")
    
    # Show top teams by final Elo
    final_elo = df_with_elo.groupby('HomeTeam')['HomeEloAfter'].last().sort_values(ascending=False)
    logger.info("\nTop 10 teams by final Elo:")
    for i, (team, elo) in enumerate(final_elo.head(10).items(), 1):
        logger.info(f"  {i:2d}. {team:30s} {elo:6.1f}")


if __name__ == '__main__':
    main()
