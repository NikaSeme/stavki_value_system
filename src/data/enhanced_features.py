"""
Enhanced Feature Extractor with Multi-Source Data.

Integrates data from:
- SportMonks: xG, lineups, injuries, statistics
- Betfair: CLV tracking, true market odds
- OpenWeatherMap: Weather conditions
- The Odds API: Multi-bookmaker odds

Produces 50+ features for ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import os

# Import clients
from .sportmonks_client import SportMonksClient, MatchFixture
from .betfair_client import BetfairClient
from .weather_client import WeatherClient
from .xg_history import XGHistory, get_xg_history
from .match_mapper import MatchMapper
from ..features.elo import EloRating
from ..utils.team_normalizer import normalize_team, get_sportmonks_id

logger = logging.getLogger(__name__)


@dataclass
class EnhancedFeatures:
    """Complete feature set for a match."""
    # Core identifiers
    event_id: str
    home_team: str
    away_team: str
    kickoff: datetime
    
    # ELO features (5)
    home_elo: float
    away_elo: float
    elo_diff: float
    home_elo_form: float  # ELO change last 5
    away_elo_form: float
    
    # Form features (10)
    home_form_points: float
    away_form_points: float
    home_goals_for_avg: float
    away_goals_for_avg: float
    home_goals_against_avg: float
    away_goals_against_avg: float
    home_clean_sheets: int
    away_clean_sheets: int
    home_win_streak: int
    away_win_streak: int
    
    # xG features (8) - FROM SPORTMONKS
    home_xg_for_avg: float
    away_xg_for_avg: float
    home_xg_against_avg: float
    away_xg_against_avg: float
    home_xg_diff: float
    away_xg_diff: float
    home_xg_overperformance: float  # Goals - xG
    away_xg_overperformance: float
    
    # Squad features (6) - FROM SPORTMONKS
    home_injuries_count: int
    away_injuries_count: int
    home_suspensions_count: int
    away_suspensions_count: int
    home_key_players_out: int  # High-value players missing
    away_key_players_out: int
    
    # Market features (8)
    best_odds_home: float
    best_odds_draw: float
    best_odds_away: float
    market_prob_home: float
    market_prob_draw: float
    market_prob_away: float
    odds_home_away_ratio: float
    market_efficiency: float
    
    # Betfair features (5) - FROM BETFAIR
    betfair_odds_home: Optional[float]
    betfair_odds_draw: Optional[float]
    betfair_odds_away: Optional[float]
    betfair_liquidity: Optional[float]
    betfair_market_confidence: Optional[float]
    
    # Weather features (5) - FROM OPENWEATHERMAP
    temperature: float
    precipitation: float
    wind_speed: float
    weather_score: float
    is_adverse_weather: int
    
    # Time features (4)
    days_since_home_match: int
    days_since_away_match: int
    hours_to_kickoff: float
    is_weekend: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def to_array(self) -> np.ndarray:
        """Convert numeric features to array for ML."""
        numeric = []
        for k, v in self.__dict__.items():
            if k in ['event_id', 'home_team', 'away_team', 'kickoff']:
                continue
            if v is None:
                numeric.append(0.0)
            elif isinstance(v, (int, float)):
                numeric.append(float(v))
        return np.array(numeric)


class EnhancedFeatureExtractor:
    """
    Multi-source feature extractor for STAVKI.
    
    Combines data from SportMonks, Betfair, OpenWeatherMap, and 
    internal ELO/form tracking to produce 50+ features.
    
    Usage:
        extractor = EnhancedFeatureExtractor(
            sportmonks_key="...",
            betfair_key="...",
            weather_key="..."
        )
        features = extractor.extract_for_match(fixture)
    """
    
    def __init__(
        self,
        sportmonks_key: Optional[str] = None,
        betfair_key: Optional[str] = None,
        weather_key: Optional[str] = None,
        state_file: Optional[str] = None
    ):
        """
        Initialize with API keys.
        
        Keys can be passed directly or loaded from environment variables:
        - SPORTMONKS_API_KEY
        - BETFAIR_APP_KEY
        - OPENWEATHER_API_KEY
        """
        # Initialize clients
        self.sportmonks = None
        self.betfair = None
        self.weather = None
        
        sm_key = sportmonks_key or os.getenv("SPORTMONKS_API_KEY")
        if sm_key:
            self.sportmonks = SportMonksClient(api_key=sm_key)
            logger.info("SportMonks client initialized")
        
        bf_key = betfair_key or os.getenv("BETFAIR_APP_KEY")
        if bf_key:
            self.betfair = BetfairClient(app_key=bf_key)
            logger.info("Betfair client initialized")
        
        wx_key = weather_key or os.getenv("OPENWEATHER_API_KEY")
        if wx_key:
            self.weather = WeatherClient(api_key=wx_key)
            logger.info("Weather client initialized")
        
        # Internal trackers
        self.elo = EloRating(k_factor=20, home_advantage=100)
        self.team_form: Dict[str, List[Dict]] = {}  # Last 5 matches per team
        self.team_last_match: Dict[str, datetime] = {}
        
        # xG history database
        self.xg_history = get_xg_history()
        
        # Match mapper for ID translation
        self.match_mapper = MatchMapper(
            sportmonks_client=self.sportmonks,
            betfair_client=self.betfair
        )
        
        # State persistence
        self.state_file = state_file
        if state_file:
            self._load_state()
        
        logger.info(f"EnhancedFeatureExtractor initialized with {self._count_sources()} data sources")
    
    def _count_sources(self) -> int:
        """Count active data sources."""
        return sum([
            self.sportmonks is not None,
            self.betfair is not None,
            self.weather is not None
        ])
    
    # =========================================================================
    # MAIN EXTRACTION
    # =========================================================================
    
    def extract_for_match(
        self,
        event_id: str,
        home_team: str,
        away_team: str,
        kickoff: datetime,
        odds_df: Optional[pd.DataFrame] = None,
        fixture_id: Optional[int] = None,  # SportMonks fixture ID
        market_id: Optional[str] = None     # Betfair market ID
    ) -> EnhancedFeatures:
        """
        Extract all features for a single match.
        
        Args:
            event_id: Unique match identifier
            home_team: Home team name
            away_team: Away team name
            kickoff: Match kickoff time
            odds_df: Optional current odds DataFrame
            fixture_id: SportMonks fixture ID for API lookups
            market_id: Betfair market ID for exchange data
            
        Returns:
            EnhancedFeatures object with 50+ features
        """
        # Base features
        features = {
            "event_id": event_id,
            "home_team": home_team,
            "away_team": away_team,
            "kickoff": kickoff
        }
        
        # ELO features
        features.update(self._extract_elo_features(home_team, away_team))
        
        # Form features
        features.update(self._extract_form_features(home_team, away_team))
        
        # xG features (SportMonks)
        features.update(self._extract_xg_features(home_team, away_team, fixture_id))
        
        # Squad features (SportMonks)
        features.update(self._extract_squad_features(fixture_id))
        
        # Market features (odds)
        features.update(self._extract_market_features(event_id, odds_df))
        
        # Betfair features
        features.update(self._extract_betfair_features(market_id))
        
        # Weather features
        features.update(self._extract_weather_features(home_team, kickoff))
        
        # Time features
        features.update(self._extract_time_features(home_team, away_team, kickoff))
        
        return EnhancedFeatures(**features)
    
    def extract_batch(
        self,
        events: pd.DataFrame,
        odds_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Extract features for multiple matches.
        
        Args:
            events: DataFrame with event_id, home_team, away_team, kickoff
            odds_df: Optional odds DataFrame
            
        Returns:
            DataFrame with all features
        """
        all_features = []
        
        for _, row in events.iterrows():
            try:
                features = self.extract_for_match(
                    event_id=row.get("event_id", str(row.name)),
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    kickoff=pd.to_datetime(row.get("kickoff", row.get("commence_time"))),
                    odds_df=odds_df,
                    fixture_id=row.get("fixture_id"),
                    market_id=row.get("market_id")
                )
                all_features.append(features.to_dict())
            except Exception as e:
                logger.warning(f"Failed to extract features for {row.get('event_id')}: {e}")
        
        return pd.DataFrame(all_features)
    
    # =========================================================================
    # FEATURE EXTRACTION METHODS
    # =========================================================================
    
    def _extract_elo_features(self, home_team: str, away_team: str) -> Dict:
        """Extract ELO rating features."""
        home_elo = self.elo.get_rating(home_team)
        away_elo = self.elo.get_rating(away_team)
        
        # Calculate form (ELO change)
        home_form = self._get_elo_form(home_team)
        away_form = self._get_elo_form(away_team)
        
        return {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "home_elo_form": home_form,
            "away_elo_form": away_form
        }
    
    def _get_elo_form(self, team: str) -> float:
        """Get ELO change over last 5 matches."""
        history = [h for h in self.elo.history if h["home_team"] == team or h["away_team"] == team]
        
        if len(history) < 2:
            return 0.0
        
        recent = history[-5:] if len(history) >= 5 else history
        
        # Calculate total ELO change
        total_change = 0.0
        for match in recent:
            if match["home_team"] == team:
                total_change += match["home_elo_after"] - match["home_elo_before"]
            else:
                total_change += match["away_elo_after"] - match["away_elo_before"]
        
        return total_change / len(recent)
    
    def _extract_form_features(self, home_team: str, away_team: str) -> Dict:
        """Extract form-based features."""
        home_form = self.team_form.get(home_team, [])
        away_form = self.team_form.get(away_team, [])
        
        return {
            "home_form_points": self._calc_form_points(home_form),
            "away_form_points": self._calc_form_points(away_form),
            "home_goals_for_avg": self._avg_stat(home_form, "goals_for"),
            "away_goals_for_avg": self._avg_stat(away_form, "goals_for"),
            "home_goals_against_avg": self._avg_stat(home_form, "goals_against"),
            "away_goals_against_avg": self._avg_stat(away_form, "goals_against"),
            "home_clean_sheets": self._count_stat(home_form, "clean_sheet"),
            "away_clean_sheets": self._count_stat(away_form, "clean_sheet"),
            "home_win_streak": self._calc_streak(home_form, "W"),
            "away_win_streak": self._calc_streak(away_form, "W")
        }
    
    def _calc_form_points(self, form: List[Dict]) -> float:
        """Calculate points from last 5 matches."""
        if not form:
            return 7.5  # Default average
        
        points = 0
        for match in form[-5:]:
            if match.get("result") == "W":
                points += 3
            elif match.get("result") == "D":
                points += 1
        
        return points
    
    def _avg_stat(self, form: List[Dict], stat: str) -> float:
        """Calculate average of a stat."""
        if not form:
            return 1.5  # Default
        
        values = [m.get(stat, 0) for m in form[-5:]]
        return sum(values) / len(values) if values else 1.5
    
    def _count_stat(self, form: List[Dict], stat: str) -> int:
        """Count occurrences of a stat."""
        if not form:
            return 0
        return sum(1 for m in form[-5:] if m.get(stat))
    
    def _calc_streak(self, form: List[Dict], result_type: str) -> int:
        """Calculate current streak of result type."""
        if not form:
            return 0
        
        streak = 0
        for match in reversed(form):
            if match.get("result") == result_type:
                streak += 1
            else:
                break
        
        return streak
    
    def _extract_xg_features(
        self,
        home_team: str,
        away_team: str,
        fixture_id: Optional[int]
    ) -> Dict:
        """Extract xG features from xG history database."""
        defaults = {
            "home_xg_for_avg": 1.3,
            "away_xg_for_avg": 1.1,
            "home_xg_against_avg": 1.1,
            "away_xg_against_avg": 1.3,
            "home_xg_diff": 0.2,
            "away_xg_diff": -0.2,
            "home_xg_overperformance": 0.0,
            "away_xg_overperformance": 0.0
        }
        
        try:
            # Normalize team names for database lookup
            home_id = normalize_team(home_team)
            away_id = normalize_team(away_team)
            
            # Get xG stats from history database
            home_stats = self.xg_history.get_team_xg_stats(home_id, last_n=5)
            away_stats = self.xg_history.get_team_xg_stats(away_id, last_n=5)
            
            return {
                "home_xg_for_avg": home_stats["xg_for_avg"],
                "away_xg_for_avg": away_stats["xg_for_avg"],
                "home_xg_against_avg": home_stats["xg_against_avg"],
                "away_xg_against_avg": away_stats["xg_against_avg"],
                "home_xg_diff": home_stats["xg_diff"],
                "away_xg_diff": away_stats["xg_diff"],
                "home_xg_overperformance": home_stats["overperformance"],
                "away_xg_overperformance": away_stats["overperformance"]
            }
        except Exception as e:
            logger.warning(f"Failed to get xG features: {e}")
            return defaults
    
    def _extract_squad_features(self, fixture_id: Optional[int]) -> Dict:
        """Extract squad features from SportMonks."""
        defaults = {
            "home_injuries_count": 0,
            "away_injuries_count": 0,
            "home_suspensions_count": 0,
            "away_suspensions_count": 0,
            "home_key_players_out": 0,
            "away_key_players_out": 0
        }
        
        if not self.sportmonks or not fixture_id:
            return defaults
        
        try:
            # Get lineups to check for missing players
            lineups = self.sportmonks.get_fixture_lineups(fixture_id)
            
            if not lineups:
                return defaults
            
            home_lineup = lineups.get("home")
            away_lineup = lineups.get("away")
            
            result = defaults.copy()
            
            # Count missing players based on typical squad size (11)
            if home_lineup:
                starting_count = len(home_lineup.starting_xi) if home_lineup.starting_xi else 0
                result["home_injuries_count"] = max(0, 11 - starting_count)
            
            if away_lineup:
                starting_count = len(away_lineup.starting_xi) if away_lineup.starting_xi else 0
                result["away_injuries_count"] = max(0, 11 - starting_count)
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to get squad features: {e}")
            return defaults
    
    def _extract_market_features(
        self,
        event_id: str,
        odds_df: Optional[pd.DataFrame]
    ) -> Dict:
        """Extract market features from odds."""
        defaults = {
            "best_odds_home": 2.0,
            "best_odds_draw": 3.5,
            "best_odds_away": 3.5,
            "market_prob_home": 0.45,
            "market_prob_draw": 0.27,
            "market_prob_away": 0.28,
            "odds_home_away_ratio": 0.57,
            "market_efficiency": 0.95
        }
        
        if odds_df is None or odds_df.empty:
            return defaults
        
        try:
            # Filter for this event
            event_odds = odds_df[odds_df["event_id"] == event_id] if "event_id" in odds_df.columns else odds_df
            
            if event_odds.empty:
                return defaults
            
            # Get best odds for each outcome
            home_odds = event_odds[event_odds["outcome"] == "home"]["outcome_price"].max() if "outcome" in event_odds.columns else 2.0
            draw_odds = event_odds[event_odds["outcome"] == "draw"]["outcome_price"].max() if "outcome" in event_odds.columns else 3.5
            away_odds = event_odds[event_odds["outcome"] == "away"]["outcome_price"].max() if "outcome" in event_odds.columns else 3.5
            
            # Calculate implied probabilities (no-vig)
            total = 1/home_odds + 1/draw_odds + 1/away_odds
            home_prob = (1/home_odds) / total
            draw_prob = (1/draw_odds) / total
            away_prob = (1/away_odds) / total
            
            # Market efficiency (inverse of overround)
            efficiency = 1.0 / total
            
            return {
                "best_odds_home": home_odds,
                "best_odds_draw": draw_odds,
                "best_odds_away": away_odds,
                "market_prob_home": home_prob,
                "market_prob_draw": draw_prob,
                "market_prob_away": away_prob,
                "odds_home_away_ratio": home_odds / (home_odds + away_odds),
                "market_efficiency": efficiency
            }
        except Exception as e:
            logger.warning(f"Failed to extract market features: {e}")
            return defaults
    
    def _extract_betfair_features(self, market_id: Optional[str]) -> Dict:
        """Extract features from Betfair Exchange."""
        defaults = {
            "betfair_odds_home": None,
            "betfair_odds_draw": None,
            "betfair_odds_away": None,
            "betfair_liquidity": None,
            "betfair_market_confidence": None
        }
        
        if not self.betfair or not market_id:
            return defaults
        
        try:
            odds = self.betfair.extract_match_odds(market_id)
            liquidity = self.betfair.get_market_liquidity(market_id)
            
            return {
                "betfair_odds_home": odds.get("home"),
                "betfair_odds_draw": odds.get("draw"),
                "betfair_odds_away": odds.get("away"),
                "betfair_liquidity": liquidity.get("total_matched"),
                "betfair_market_confidence": 1.0 if liquidity.get("is_liquid") else 0.5
            }
        except Exception as e:
            logger.warning(f"Failed to get Betfair features: {e}")
            return defaults
    
    def _extract_weather_features(
        self,
        home_team: str,
        kickoff: datetime
    ) -> Dict:
        """Extract weather features."""
        defaults = {
            "temperature": 15.0,
            "precipitation": 0.0,
            "wind_speed": 0.0,
            "weather_score": 0.8,
            "is_adverse_weather": 0
        }
        
        if not self.weather:
            return defaults
        
        try:
            features = self.weather.get_weather_features(
                home_team=home_team,
                kickoff_time=kickoff
            )
            
            return {
                "temperature": features.get("temperature", 15.0),
                "precipitation": features.get("precipitation", 0.0),
                "wind_speed": features.get("wind_speed", 0.0),
                "weather_score": features.get("weather_score", 0.8),
                "is_adverse_weather": 1 if features.get("weather_score", 1.0) < 0.6 else 0
            }
        except Exception as e:
            logger.warning(f"Failed to get weather features: {e}")
            return defaults
    
    def _extract_time_features(
        self,
        home_team: str,
        away_team: str,
        kickoff: datetime
    ) -> Dict:
        """Extract time-based features."""
        now = datetime.now()
        
        # Days since last match
        home_last = self.team_last_match.get(home_team, now - timedelta(days=7))
        away_last = self.team_last_match.get(away_team, now - timedelta(days=7))
        
        return {
            "days_since_home_match": (kickoff - home_last).days,
            "days_since_away_match": (kickoff - away_last).days,
            "hours_to_kickoff": max(0, (kickoff - now).total_seconds() / 3600),
            "is_weekend": 1 if kickoff.weekday() >= 5 else 0
        }
    
    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    def update_after_match(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        match_date: datetime
    ):
        """
        Update internal state after a match.
        
        Call this after each match to maintain accurate ELO and form.
        """
        # Determine result
        if home_goals > away_goals:
            result = "H"
            home_result, away_result = "W", "L"
        elif home_goals < away_goals:
            result = "A"
            home_result, away_result = "L", "W"
        else:
            result = "D"
            home_result, away_result = "D", "D"
        
        # Update ELO
        self.elo.update(home_team, away_team, result, match_date)
        
        # Update form
        self._update_form(home_team, home_goals, away_goals, home_result)
        self._update_form(away_team, away_goals, home_goals, away_result)
        
        # Update last match dates
        self.team_last_match[home_team] = match_date
        self.team_last_match[away_team] = match_date
        
        # Persist state
        if self.state_file:
            self._save_state()
    
    def _update_form(self, team: str, goals_for: int, goals_against: int, result: str):
        """Update team form."""
        if team not in self.team_form:
            self.team_form[team] = []
        
        self.team_form[team].append({
            "goals_for": goals_for,
            "goals_against": goals_against,
            "result": result,
            "clean_sheet": goals_against == 0
        })
        
        # Keep only last 10 matches
        self.team_form[team] = self.team_form[team][-10:]
    
    def _save_state(self):
        """Save state to file."""
        import json
        
        state = {
            "elo_ratings": self.elo.ratings,
            "elo_history": self.elo.history[-100:],  # Last 100 matches
            "team_form": self.team_form,
            "team_last_match": {k: v.isoformat() for k, v in self.team_last_match.items()}
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f)
    
    def _load_state(self):
        """Load state from file."""
        import json
        from pathlib import Path
        
        if not Path(self.state_file).exists():
            return
        
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            
            self.elo.ratings = state.get("elo_ratings", {})
            self.elo.history = state.get("elo_history", [])
            self.team_form = state.get("team_form", {})
            self.team_last_match = {
                k: datetime.fromisoformat(v) 
                for k, v in state.get("team_last_match", {}).items()
            }
            
            logger.info(f"Loaded state: {len(self.elo.ratings)} teams, {len(self.elo.history)} matches")
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
    
    # =========================================================================
    # FEATURE LIST
    # =========================================================================
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get list of all feature names."""
        return [
            # ELO (5)
            "home_elo", "away_elo", "elo_diff", "home_elo_form", "away_elo_form",
            # Form (10)
            "home_form_points", "away_form_points", 
            "home_goals_for_avg", "away_goals_for_avg",
            "home_goals_against_avg", "away_goals_against_avg",
            "home_clean_sheets", "away_clean_sheets",
            "home_win_streak", "away_win_streak",
            # xG (8)
            "home_xg_for_avg", "away_xg_for_avg",
            "home_xg_against_avg", "away_xg_against_avg",
            "home_xg_diff", "away_xg_diff",
            "home_xg_overperformance", "away_xg_overperformance",
            # Squad (6)
            "home_injuries_count", "away_injuries_count",
            "home_suspensions_count", "away_suspensions_count",
            "home_key_players_out", "away_key_players_out",
            # Market (8)
            "best_odds_home", "best_odds_draw", "best_odds_away",
            "market_prob_home", "market_prob_draw", "market_prob_away",
            "odds_home_away_ratio", "market_efficiency",
            # Betfair (5)
            "betfair_odds_home", "betfair_odds_draw", "betfair_odds_away",
            "betfair_liquidity", "betfair_market_confidence",
            # Weather (5)
            "temperature", "precipitation", "wind_speed",
            "weather_score", "is_adverse_weather",
            # Time (4)
            "days_since_home_match", "days_since_away_match",
            "hours_to_kickoff", "is_weekend"
        ]  # Total: 51 features


# Convenience function
def create_extractor(
    sportmonks_key: Optional[str] = None,
    betfair_key: Optional[str] = None,
    weather_key: Optional[str] = None,
    state_file: str = "data/feature_state.json"
) -> EnhancedFeatureExtractor:
    """Create an enhanced feature extractor with optional key overrides."""
    return EnhancedFeatureExtractor(
        sportmonks_key=sportmonks_key,
        betfair_key=betfair_key,
        weather_key=weather_key,
        state_file=state_file
    )
