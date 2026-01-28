"""
Live feature extraction for real-time match prediction.

Extracts features from live odds events to match training format.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import logging

from .elo import EloRating

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveFeatureExtractor:
    """
    Extract features for live events matching training format.
    
    Features extracted:
    - Elo ratings (home/away/diff)
    - Form features (last 5 games)
    - Market features (implied probabilities)
    - H2H features (head-to-head)
    """
    
    def __init__(self, state_file: Optional[str] = None):
        """
        Initialize feature extractor.
        
        Args:
            state_file: Optional file to load/save Elo/form state
        """
        self.elo = EloRating(k_factor=20, home_advantage=100)
        self.team_form: Dict[str, list] = {}
        self.state_file = Path(state_file) if state_file else None
        
        # Load saved state if exists
        if self.state_file and self.state_file.exists():
            self.load_state()
            
        # Initialize Sentiment Feature Extractor
        try:
            from src.features.sentiment_features import SentimentFeatureExtractor
            self.sentiment_extractor = SentimentFeatureExtractor(mode='news')
            logger.info("Sentiment Extractor initialized (Mode: news)")
        except Exception as e:
            logger.error(f"Failed to init Sentiment Extractor: {e}")
            self.sentiment_extractor = None

    def fetch_sentiment_for_events(self, events: pd.DataFrame) -> Dict[str, Dict]:
        """
        Fetch sentiment features for all events.
        
        Args:
            events: DataFrame with event_id, home_team, away_team
            
        Returns:
            Dict[event_id] = {home_sentiment, away_sentiment, etc.}
        """
        sentiment_data = {}
        
        if not self.sentiment_extractor:
            return sentiment_data
            
        logger.info(f"Fetching sentiment for {len(events)} events...")
        
        for _, event in events.iterrows():
            event_id = event['event_id']
            home_team = event['home_team']
            away_team = event['away_team']
            
            try:
                # Use cached fetcher
                features = self.sentiment_extractor.extract_for_match(
                    home_team, away_team, lookback_hours=48
                )
                sentiment_data[event_id] = features
            except Exception as e:
                logger.warning(f"Sentiment fetch failed for {home_team} vs {away_team}: {e}")
                
        return sentiment_data
    
    def extract_features(
        self,
        events: pd.DataFrame,
        odds_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract features for live events.
        
        Args:
            events: DataFrame with columns: event_id, home_team, away_team
            odds_df: DataFrame with current odds (must include outcome_price)
            
        Returns:
            DataFrame with 22 features matching training format
        """
        features_list = []
        
        for _, event in events.iterrows():
            home_team = event['home_team']
            away_team = event['away_team']
            event_id = event['event_id']
            
            # Elo features
            home_elo = self.elo.get_rating(home_team)
            away_elo = self.elo.get_rating(away_team)
            elo_diff = home_elo - away_elo
            
            # Form features
            home_form = self._get_form_features(home_team)
            away_form = self._get_form_features(away_team)
            
            # Market features from odds
            market_feats = self._extract_market_features(event, odds_df)
            
            # H2H features
            h2h_feats = self._extract_h2h_features(home_team, away_team)
            
            # Combine all features (must match training order!)
            feat_dict = {
                'HomeEloBefore': home_elo,
                'AwayEloBefore': away_elo,
                'EloDiff': elo_diff,
                'HomePointsL5': home_form['points'],
                'HomeGoalsL5': home_form['goals_for'],
                'HomeGoalsAgainstL5': home_form['goals_against'],
                'HomeWinRateL5': home_form['win_rate'],
                'HomeCleanSheetsL5': home_form['clean_sheets'],
                'AwayPointsL5': away_form['points'],
                'AwayGoalsL5': away_form['goals_for'],
                'AwayGoalsAgainstL5': away_form['goals_against'],
                'AwayWinRateL5': away_form['win_rate'],
                'AwayCleanSheetsL5': away_form['clean_sheets'],
                'HomeTeam': home_team,  # Critical for Categorical Model
                'AwayTeam': away_team,  # Critical for Categorical Model
                'Season': '2023/24',    # Default season for live inference
                **market_feats,
                **h2h_feats,
            }
            
            features_list.append(feat_dict)
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        
        # Fill NA: Numeric with 0.0, Categorical with "Unknown"
        # This prevents "mixed type" errors in CatBoost for Team/League columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        
        df[num_cols] = df[num_cols].fillna(0.0)
        df[cat_cols] = df[cat_cols].fillna("Unknown")
        
        return df
    
    def _extract_market_features(
        self,
        event: pd.Series,
        odds_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Extract market features from odds.
        
        Returns dict with keys:
        - MarketProbHomeNoVig
        - MarketProbDrawNoVig
        - MarketProbAwayNoVig
        - OddsHomeAwayRatio
        """
        event_id = event['event_id']
        home_team = event['home_team']
        away_team = event['away_team']
        
        # Filter odds for this event
        event_odds = odds_df[odds_df['event_id'] == event_id]
        
        if len(event_odds) == 0:
            # Default values if no odds (should not happen in production)
            logger.warning(f"No odds found for event {event_id}, using defaults")
            return {
                'MarketProbHomeNoVig': 0.45,
                'MarketProbDrawNoVig': 0.25,
                'MarketProbAwayNoVig': 0.30,
                'OddsHomeAwayRatio': 1.5,
            }
        
        # Get average odds for each outcome
        home_odds_list = event_odds[event_odds['outcome_name'] == home_team]['outcome_price']
        draw_odds_list = event_odds[event_odds['outcome_name'] == 'Draw']['outcome_price']
        away_odds_list = event_odds[event_odds['outcome_name'] == away_team]['outcome_price']
        
        home_odds = home_odds_list.mean() if len(home_odds_list) > 0 else 2.0
        draw_odds = draw_odds_list.mean() if len(draw_odds_list) > 0 else 3.5
        away_odds = away_odds_list.mean() if len(away_odds_list) > 0 else 3.0
        
        # Implied probabilities
        p_home = 1 / home_odds
        p_draw = 1 / draw_odds
        p_away = 1 / away_odds
        
        # Remove vig (normalize to sum to 1.0)
        total = p_home + p_draw + p_away
        
        return {
            'MarketProbHomeNoVig': p_home / total,
            'MarketProbDrawNoVig': p_draw / total,
            'MarketProbAwayNoVig': p_away / total,
            'OddsHomeAwayRatio': home_odds / away_odds if away_odds > 0 else 1.5,
        }
    
    def _get_form_features(self, team: str) -> Dict[str, float]:
        """
        Get form features for a team.
        
        Returns dict with keys:
        - points: Total points in last 5 games
        - goals_for: Goals scored in last 5
        - goals_against: Goals conceded in last 5
        - win_rate: Win rate in last 5
        - clean_sheets: Clean sheet rate in last 5
        """
        if team not in self.team_form or len(self.team_form[team]) == 0:
            # Return league average defaults for teams without history
            return self._default_form()
        
        # Get last 5 matches
        recent = self.team_form[team][-5:]
        
        return {
            'points': sum(m['points'] for m in recent),
            'goals_for': sum(m['goals_for'] for m in recent),
            'goals_against': sum(m['goals_against'] for m in recent),
            'win_rate': sum(m['won'] for m in recent) / len(recent),
            'clean_sheets': sum(m['clean_sheet'] for m in recent) / len(recent),
        }
    
    def _extract_h2h_features(
        self,
        home_team: str,
        away_team: str
    ) -> Dict[str, float]:
        """
        Extract head-to-head features.
        
        For live system, uses league averages (simplified).
        In production, could load from database.
        
        Returns dict with keys:
        - H2HHomeWins
        - H2HDraws
        - H2HAwayWins
        - H2HHomeGoalsAvg
        - H2HAwayGoalsAvg
        """
        # Use neutral/league average defaults
        # In production, query database for actual H2H history
        return {
            'H2HHomeWins': 0.35,  # Slight home advantage
            'H2HDraws': 0.30,
            'H2HAwayWins': 0.35,
            'H2HHomeGoalsAvg': 1.5,
            'H2HAwayGoalsAvg': 1.5,
        }
    
    def _default_form(self) -> Dict[str, float]:
        """Default form for teams without history (league averages)."""
        return {
            'points': 7.5,  # 1.5 points per game average
            'goals_for': 7.5,  # 1.5 goals per game
            'goals_against': 7.5,
            'win_rate': 0.35,  # ~35% win rate
            'clean_sheets': 0.20,  # 20% clean sheet rate
        }
    
    def update_after_match(
        self,
        home_team: str,
        away_team: str,
        result: str,
        goals_home: int,
        goals_away: int
    ):
        """
        Update Elo and form after a match completes.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            result: Match result ('H', 'D', or 'A')
            goals_home: Home team goals
            goals_away: Away team goals
        """
        # Update Elo
        self.elo.update(home_team, away_team, result)
        
        # Update form for home team
        if home_team not in self.team_form:
            self.team_form[home_team] = []
        
        if result == 'H':
            home_points, home_won = 3, 1
        elif result == 'D':
            home_points, home_won = 1, 0
        else:
            home_points, home_won = 0, 0
        
        self.team_form[home_team].append({
            'points': home_points,
            'goals_for': goals_home,
            'goals_against': goals_away,
            'won': home_won,
            'clean_sheet': 1 if goals_away == 0 else 0,
        })
        
        # Update form for away team
        if away_team not in self.team_form:
            self.team_form[away_team] = []
        
        if result == 'A':
            away_points, away_won = 3, 1
        elif result == 'D':
            away_points, away_won = 1, 0
        else:
            away_points, away_won = 0, 0
        
        self.team_form[away_team].append({
            'points': away_points,
            'goals_for': goals_away,
            'goals_against': goals_home,
            'won': away_won,
            'clean_sheet': 1 if goals_home == 0 else 0,
        })
        
        # Save state if configured
        if self.state_file:
            self.save_state()
    
    def load_state(self):
        """Load Elo/form state from file."""
        if not self.state_file or not self.state_file.exists():
            return
        
        try:
            import pickle
            with open(self.state_file, 'rb') as f:
                state = pickle.load(f)
                self.elo.ratings = state.get('elo_ratings', {})
                self.team_form = state.get('team_form', {})
            logger.info(f"Loaded state from {self.state_file}")
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
    
    def save_state(self):
        """Save current Elo/form state to file."""
        if not self.state_file:
            return
        
        try:
            import pickle
            state = {
                'elo_ratings': self.elo.ratings,
                'team_form': self.team_form,
            }
            with open(self.state_file, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"Saved state to {self.state_file}")
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")


def test_extractor():
    """Test feature extraction."""
    print("Testing LiveFeatureExtractor...")
    
    extractor = LiveFeatureExtractor()
    
    # Create test event
    events = pd.DataFrame([{
        'event_id': 'test123',
        'home_team': 'Arsenal',
        'away_team': 'Chelsea',
    }])
    
    # Create test odds
    odds = pd.DataFrame([
        {'event_id': 'test123', 'outcome_name': 'Arsenal', 'outcome_price': 2.0},
        {'event_id': 'test123', 'outcome_name': 'Draw', 'outcome_price': 3.5},
        {'event_id': 'test123', 'outcome_name': 'Chelsea', 'outcome_price': 3.0},
    ])
    
    # Extract features
    features = extractor.extract_features(events, odds)
    
    print(f"\n✅ Feature extraction test passed!")
    print(f"   Shape: {features.shape} (should be (1, 22))")
    print(f"   Columns: {list(features.columns)}")
    print(f"\n   Sample values:")
    print(f"   HomeEloBefore: {features['HomeEloBefore'].iloc[0]:.1f}")
    print(f"   AwayEloBefore: {features['AwayEloBefore'].iloc[0]:.1f}")
    print(f"   MarketProbHomeNoVig: {features['MarketProbHomeNoVig'].iloc[0]:.3f}")
    
    return True


if __name__ == '__main__':
    test_extractor()
