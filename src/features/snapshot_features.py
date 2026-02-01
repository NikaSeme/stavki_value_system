"""
Snapshot Feature Engineering (Task 4)

Builds features from snapshot data with strict time-safety guarantees.
All features are computed using ONLY data available at snapshot_time.
OPTIMIZED: Pre-computes ELO ratings once during initialization.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SnapshotFeatureBuilder:
    """
    Builds features for a match at a specific snapshot time.
    
    OPTIMIZED: Pre-computes ELO ratings for all matches during init.
    This avoids O(n²) complexity when building features for all matches.
    
    Guarantees:
    - All features use only data available BEFORE snapshot_time
    - Elo ratings computed from matches before snapshot_time
    - Form computed from matches before snapshot_time
    - No future leakage
    """
    
    def __init__(
        self, 
        historical_matches: pd.DataFrame,
        k_factor: float = 20.0,
        home_advantage: float = 100.0,
        initial_rating: float = 1500.0
    ):
        """
        Initialize with historical match data and pre-compute ELO.
        
        Args:
            historical_matches: DataFrame with columns:
                Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, League
        """
        self.matches = historical_matches.copy()
        self.matches['Date'] = pd.to_datetime(self.matches['Date'])
        self.matches = self.matches.sort_values('Date').reset_index(drop=True)
        
        # Precompute team lists
        self.all_teams = set(self.matches['HomeTeam'].unique()) | set(self.matches['AwayTeam'].unique())
        
        # Pre-compute ELO ratings for all matches
        logger.info(f"Pre-computing ELO ratings for {len(self.matches)} matches...")
        self._precompute_elo(k_factor, home_advantage, initial_rating)
        
        logger.info(f"FeatureBuilder initialized with {len(self.matches)} matches")
    
    def _precompute_elo(
        self,
        k_factor: float,
        home_advantage: float,
        initial_rating: float
    ):
        """Pre-compute ELO ratings chronologically and store in DataFrame."""
        ratings = {t: initial_rating for t in self.all_teams}
        
        elo_home_before = []
        elo_away_before = []
        
        for idx, match in self.matches.iterrows():
            home = match['HomeTeam']
            away = match['AwayTeam']
            result = match['FTR']
            
            # Store BEFORE ratings
            elo_home_before.append(ratings[home])
            elo_away_before.append(ratings[away])
            
            # Expected scores
            home_adj = ratings[home] + home_advantage
            away_adj = ratings[away]
            exp_home = 1 / (1 + 10 ** ((away_adj - home_adj) / 400))
            exp_away = 1 - exp_home
            
            # Actual scores
            if result == 'H':
                actual_home, actual_away = 1.0, 0.0
            elif result == 'A':
                actual_home, actual_away = 0.0, 1.0
            else:
                actual_home, actual_away = 0.5, 0.5
            
            # Update ratings
            ratings[home] += k_factor * (actual_home - exp_home)
            ratings[away] += k_factor * (actual_away - exp_away)
        
        # Store in DataFrame
        self.matches['_elo_home'] = elo_home_before
        self.matches['_elo_away'] = elo_away_before
        
        # Create lookup dict: (date, home_team, away_team) -> (elo_home, elo_away)
        self._elo_lookup = {}
        for _, row in self.matches.iterrows():
            key = (row['Date'], row['HomeTeam'], row['AwayTeam'])
            self._elo_lookup[key] = (row['_elo_home'], row['_elo_away'])
    
    def get_precomputed_elo(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime
    ) -> Tuple[float, float]:
        """Get pre-computed ELO for a specific match."""
        key = (match_date, home_team, away_team)
        if key in self._elo_lookup:
            return self._elo_lookup[key]
        return 1500.0, 1500.0  # Default
    
    def compute_form_at_time(
        self,
        team: str,
        as_of: datetime,
        n_games: int = 5
    ) -> Dict[str, float]:
        """
        Compute form features for team using only matches before as_of.
        """
        # Get team's past matches (vectorized filtering)
        mask = (
            (self.matches['Date'] < as_of) & 
            ((self.matches['HomeTeam'] == team) | (self.matches['AwayTeam'] == team))
        )
        past = self.matches[mask].tail(n_games)
        
        if past.empty:
            return {'pts': 1.2, 'gf': 1.3, 'ga': 1.3}
        
        pts = []
        gf = []
        ga = []
        
        for _, match in past.iterrows():
            is_home = match['HomeTeam'] == team
            
            if is_home:
                goals_for = match['FTHG']
                goals_against = match['FTAG']
                points = 3 if match['FTR'] == 'H' else (1 if match['FTR'] == 'D' else 0)
            else:
                goals_for = match['FTAG']
                goals_against = match['FTHG']
                points = 3 if match['FTR'] == 'A' else (1 if match['FTR'] == 'D' else 0)
            
            pts.append(points)
            gf.append(goals_for)
            ga.append(goals_against)
        
        return {'pts': np.mean(pts), 'gf': np.mean(gf), 'ga': np.mean(ga)}
    
    def compute_rest_days(self, team: str, as_of: datetime) -> float:
        """Compute days since team's last match before as_of."""
        mask = (
            (self.matches['Date'] < as_of) & 
            ((self.matches['HomeTeam'] == team) | (self.matches['AwayTeam'] == team))
        )
        past = self.matches[mask]
        
        if past.empty:
            return 7.0
        
        last_match_date = past['Date'].max()
        rest_days = (as_of - last_match_date).days
        return float(min(rest_days, 30))
    
    def build_football_features(
        self,
        home_team: str,
        away_team: str,
        snapshot_time: datetime,
        league: str
    ) -> Dict[str, Any]:
        """Build all football features for a match at snapshot_time."""
        # Use pre-computed ELO
        elo_home, elo_away = self.get_precomputed_elo(home_team, away_team, snapshot_time)
        
        # Form (still needs to be computed but is fast)
        form_home = self.compute_form_at_time(home_team, snapshot_time, n_games=5)
        form_away = self.compute_form_at_time(away_team, snapshot_time, n_games=5)
        
        # Rest days
        rest_home = self.compute_rest_days(home_team, snapshot_time)
        rest_away = self.compute_rest_days(away_team, snapshot_time)
        
        return {
            'elo_home': elo_home,
            'elo_away': elo_away,
            'elo_diff': elo_home - elo_away,
            'form_pts_home_l5': form_home['pts'],
            'form_pts_away_l5': form_away['pts'],
            'form_gf_home_l5': form_home['gf'],
            'form_gf_away_l5': form_away['gf'],
            'form_ga_home_l5': form_home['ga'],
            'form_ga_away_l5': form_away['ga'],
            'rest_days_home': rest_home,
            'rest_days_away': rest_away,
            'league': league,
            'home_team': home_team,
            'away_team': away_team,
        }


def build_market_features(
    odds_home: float,
    odds_draw: float, 
    odds_away: float,
    all_home_odds: Optional[list] = None,
    all_draw_odds: Optional[list] = None,
    all_away_odds: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Build market features from odds.
    
    Args:
        odds_home: Best home odds
        odds_draw: Best draw odds
        odds_away: Best away odds
        all_*_odds: Optional list of all bookmaker odds for dispersion
    """
    # Implied probabilities
    implied_home = 1 / odds_home
    implied_draw = 1 / odds_draw
    implied_away = 1 / odds_away
    
    # Market overround
    overround = implied_home + implied_draw + implied_away
    
    # No-vig (fair) probabilities
    no_vig_home = implied_home / overround
    no_vig_draw = implied_draw / overround
    no_vig_away = implied_away / overround
    
    # Line dispersion (if multiple books available)
    if all_home_odds and len(all_home_odds) > 1:
        disp_home = np.std(all_home_odds)
        disp_draw = np.std(all_draw_odds) if all_draw_odds else 0.0
        disp_away = np.std(all_away_odds) if all_away_odds else 0.0
        book_count = len(all_home_odds)
    else:
        disp_home = 0.0
        disp_draw = 0.0
        disp_away = 0.0
        book_count = 1
    
    return {
        'odds_home': odds_home,
        'odds_draw': odds_draw,
        'odds_away': odds_away,
        'implied_home': implied_home,
        'implied_draw': implied_draw,
        'implied_away': implied_away,
        'no_vig_home': no_vig_home,
        'no_vig_draw': no_vig_draw,
        'no_vig_away': no_vig_away,
        'market_overround': overround,
        'line_dispersion_home': disp_home,
        'line_dispersion_draw': disp_draw,
        'line_dispersion_away': disp_away,
        'book_count': book_count,
    }


def make_features(
    home_team: str,
    away_team: str,
    league: str,
    snapshot_time: datetime,
    odds_home: float,
    odds_draw: float,
    odds_away: float,
    feature_builder: SnapshotFeatureBuilder,
    all_home_odds: Optional[list] = None,
    all_draw_odds: Optional[list] = None,
    all_away_odds: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Build complete feature set for a match at snapshot_time.
    
    This is the main function that combines all feature groups.
    Output columns match FEATURE_ORDER from snapshot_config.py.
    """
    # Football features
    football = feature_builder.build_football_features(
        home_team=home_team,
        away_team=away_team,
        snapshot_time=snapshot_time,
        league=league
    )
    
    # Market features
    market = build_market_features(
        odds_home=odds_home,
        odds_draw=odds_draw,
        odds_away=odds_away,
        all_home_odds=all_home_odds,
        all_draw_odds=all_draw_odds,
        all_away_odds=all_away_odds,
    )
    
    # Combine
    features = {**market, **football}
    
    return features


def print_feature_contract():
    """Print the feature contract for documentation."""
    from src.config.snapshot_config import FEATURE_ORDER, FEATURE_COLUMNS
    
    print("\n" + "=" * 50)
    print("FEATURE CONTRACT (v1.0)")
    print("=" * 50)
    print(f"\nTotal features: {len(FEATURE_ORDER)}")
    print("\nColumns (in order):")
    for i, col in enumerate(FEATURE_ORDER, 1):
        dtype = FEATURE_COLUMNS[col].__name__
        print(f"  {i:2}. {col:<25} ({dtype})")


if __name__ == "__main__":
    print_feature_contract()
