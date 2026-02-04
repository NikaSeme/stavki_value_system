"""
API Configuration for STAVKI Value System.

Centralized configuration for all external API services.
All API keys are loaded from environment variables for security.
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class SportMonksConfig:
    """SportMonks Football API configuration."""
    api_key: str
    base_url: str = "https://api.sportmonks.com/v3/football"
    rate_limit_per_minute: int = 180  # European Advanced tier
    timeout: int = 30
    
    # Included leagues in European Advanced
    EUROPEAN_LEAGUES = {
        "EPL": 8,           # Premier League
        "LA_LIGA": 564,     # La Liga
        "BUNDESLIGA": 82,   # Bundesliga
        "SERIE_A": 384,     # Serie A
        "LIGUE_1": 301,     # Ligue 1
        "CHAMPIONSHIP": 9,  # Championship
        "EREDIVISIE": 72,   # Eredivisie
        "PRIMEIRA": 462,    # Primeira Liga
        "SCOTTISH": 501,    # Scottish Premiership
        "BELGIAN": 208,     # Belgian Pro League
    }


@dataclass
class BetfairConfig:
    """Betfair Exchange API configuration."""
    app_key: str
    base_url: str = "https://api.betfair.com/exchange"
    timeout: int = 30
    
    # Football competition IDs
    FOOTBALL_EVENT_TYPE_ID = "1"
    
    # Market types we care about
    MARKET_TYPES = ["MATCH_ODDS", "OVER_UNDER_25"]


@dataclass
class OpenWeatherConfig:
    """OpenWeatherMap API configuration."""
    api_key: str
    base_url: str = "https://api.openweathermap.org/data/2.5"
    timeout: int = 15
    
    # Stadium coordinates for major venues
    STADIUM_COORDS = {
        # Premier League
        "Old Trafford": (53.4631, -2.2913),
        "Anfield": (53.4308, -2.9608),
        "Emirates Stadium": (51.5549, -0.1084),
        "Stamford Bridge": (51.4817, -0.1910),
        "Etihad Stadium": (53.4831, -2.2004),
        "Tottenham Hotspur Stadium": (51.6042, -0.0662),
        # Add more as needed
    }


@dataclass
class TheOddsApiConfig:
    """The Odds API configuration (existing)."""
    api_key: str
    base_url: str = "https://api.the-odds-api.com/v4"
    timeout: int = 30


class APIConfig:
    """
    Centralized API configuration manager.
    
    Usage:
        config = APIConfig.load()
        sportmonks = config.sportmonks
        betfair = config.betfair
    """
    
    def __init__(
        self,
        sportmonks: Optional[SportMonksConfig] = None,
        betfair: Optional[BetfairConfig] = None,
        openweather: Optional[OpenWeatherConfig] = None,
        odds_api: Optional[TheOddsApiConfig] = None
    ):
        self.sportmonks = sportmonks
        self.betfair = betfair
        self.openweather = openweather
        self.odds_api = odds_api
    
    @classmethod
    def load(cls, env_file: Optional[Path] = None) -> "APIConfig":
        """
        Load configuration from environment variables.
        
        Environment variables:
            SPORTMONKS_API_KEY
            BETFAIR_APP_KEY
            OPENWEATHER_API_KEY
            ODDS_API_KEY
        """
        # Try loading from .env file if provided
        if env_file and env_file.exists():
            cls._load_env_file(env_file)
        
        # SportMonks
        sportmonks_key = os.getenv("SPORTMONKS_API_KEY")
        sportmonks = SportMonksConfig(api_key=sportmonks_key) if sportmonks_key else None
        
        # Betfair
        betfair_key = os.getenv("BETFAIR_APP_KEY")
        betfair = BetfairConfig(app_key=betfair_key) if betfair_key else None
        
        # OpenWeather
        openweather_key = os.getenv("OPENWEATHER_API_KEY")
        openweather = OpenWeatherConfig(api_key=openweather_key) if openweather_key else None
        
        # The Odds API
        odds_key = os.getenv("ODDS_API_KEY")
        odds_api = TheOddsApiConfig(api_key=odds_key) if odds_key else None
        
        config = cls(
            sportmonks=sportmonks,
            betfair=betfair,
            openweather=openweather,
            odds_api=odds_api
        )
        
        # Log what's configured
        logger.info("API Configuration loaded:")
        logger.info(f"  SportMonks: {'✓' if sportmonks else '✗'}")
        logger.info(f"  Betfair: {'✓' if betfair else '✗'}")
        logger.info(f"  OpenWeather: {'✓' if openweather else '✗'}")
        logger.info(f"  The Odds API: {'✓' if odds_api else '✗'}")
        
        return config
    
    @classmethod
    def from_keys(
        cls,
        sportmonks_key: Optional[str] = None,
        betfair_key: Optional[str] = None,
        openweather_key: Optional[str] = None,
        odds_api_key: Optional[str] = None
    ) -> "APIConfig":
        """Create config directly from API keys."""
        return cls(
            sportmonks=SportMonksConfig(api_key=sportmonks_key) if sportmonks_key else None,
            betfair=BetfairConfig(app_key=betfair_key) if betfair_key else None,
            openweather=OpenWeatherConfig(api_key=openweather_key) if openweather_key else None,
            odds_api=TheOddsApiConfig(api_key=odds_api_key) if odds_api_key else None
        )
    
    @staticmethod
    def _load_env_file(env_file: Path):
        """Load environment variables from .env file."""
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")
    
    def validate(self) -> dict:
        """Validate all configurations and return status."""
        return {
            "sportmonks": {
                "configured": self.sportmonks is not None,
                "valid": self._validate_sportmonks()
            },
            "betfair": {
                "configured": self.betfair is not None,
                "valid": self._validate_betfair()
            },
            "openweather": {
                "configured": self.openweather is not None,
                "valid": self._validate_openweather()
            },
            "odds_api": {
                "configured": self.odds_api is not None,
                "valid": self._validate_odds_api()
            }
        }
    
    def _validate_sportmonks(self) -> bool:
        """Validate SportMonks API key."""
        if not self.sportmonks:
            return False
        # Key should be 60 chars
        return len(self.sportmonks.api_key) >= 50
    
    def _validate_betfair(self) -> bool:
        """Validate Betfair app key."""
        if not self.betfair:
            return False
        return len(self.betfair.app_key) >= 10
    
    def _validate_openweather(self) -> bool:
        """Validate OpenWeatherMap API key."""
        if not self.openweather:
            return False
        return len(self.openweather.api_key) == 32
    
    def _validate_odds_api(self) -> bool:
        """Validate The Odds API key."""
        if not self.odds_api:
            return False
        return len(self.odds_api.api_key) >= 20


# Default configuration instance
_config: Optional[APIConfig] = None


def get_config() -> APIConfig:
    """Get the global API configuration."""
    global _config
    if _config is None:
        _config = APIConfig.load()
    return _config


def init_config(
    sportmonks_key: Optional[str] = None,
    betfair_key: Optional[str] = None,
    openweather_key: Optional[str] = None,
    odds_api_key: Optional[str] = None
) -> APIConfig:
    """Initialize API configuration with provided keys."""
    global _config
    _config = APIConfig.from_keys(
        sportmonks_key=sportmonks_key,
        betfair_key=betfair_key,
        openweather_key=openweather_key,
        odds_api_key=odds_api_key
    )
    return _config
