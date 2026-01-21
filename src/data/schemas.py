"""
Data schemas for football match records.
Defines raw and normalized match data structures.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


# Column name mappings from football-data.co.uk format
FOOTBALL_DATA_COLUMNS = {
    "Div": "league",
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "B365H": "odds_1",
    "B365D": "odds_x",
    "B365A": "odds_2",
}

# Required columns for a valid match record
REQUIRED_COLUMNS = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]

# Odds columns (optional)
ODDS_COLUMNS = ["B365H", "B365D", "B365A"]


@dataclass
class RawMatchRecord:
    """Raw match record from football-data CSV format."""
    
    Div: str
    Date: str
    HomeTeam: str
    AwayTeam: str
    FTHG: int
    FTAG: int
    B365H: Optional[float] = None
    B365D: Optional[float] = None
    B365A: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Validate raw data types."""
        if not isinstance(self.Div, str):
            raise ValueError(f"Div must be string, got {type(self.Div)}")
        if not isinstance(self.HomeTeam, str):
            raise ValueError(f"HomeTeam must be string, got {type(self.HomeTeam)}")
        if not isinstance(self.AwayTeam, str):
            raise ValueError(f"AwayTeam must be string, got {type(self.AwayTeam)}")


@dataclass
class NormalizedMatch:
    """Normalized match record with standard column names."""
    
    date: datetime
    league: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    odds_1: Optional[float] = None  # Home win
    odds_x: Optional[float] = None  # Draw
    odds_2: Optional[float] = None  # Away win
    
    def __post_init__(self) -> None:
        """Validate normalized data."""
        if not isinstance(self.date, datetime):
            raise ValueError(f"date must be datetime, got {type(self.date)}")
        
        if self.home_goals < 0:
            raise ValueError(f"home_goals cannot be negative: {self.home_goals}")
        
        if self.away_goals < 0:
            raise ValueError(f"away_goals cannot be negative: {self.away_goals}")
        
        # Validate odds if present
        for odds, name in [
            (self.odds_1, "odds_1"),
            (self.odds_x, "odds_x"),
            (self.odds_2, "odds_2")
        ]:
            if odds is not None and odds < 1.0:
                raise ValueError(f"{name} must be >= 1.0 (got {odds})")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export."""
        return {
            "date": self.date.strftime("%Y-%m-%d"),
            "league": self.league,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
            "odds_1": self.odds_1,
            "odds_x": self.odds_x,
            "odds_2": self.odds_2,
        }
    
    def match_key(self) -> tuple:
        """Unique key for deduplication (date, home_team, away_team)."""
        return (self.date.strftime("%Y-%m-%d"), self.home_team, self.away_team)
