"""
Data ingestion package for STAVKI betting system.
Handles loading and normalization of football match data,
and API clients for external data sources.
"""

from .ingestion import ingest_directory, load_csv, normalize_matches
from .schemas import NormalizedMatch, RawMatchRecord

# API Clients
from .sportmonks_client import SportMonksClient
from .betfair_client import BetfairClient
from .weather_client import WeatherClient
from .enhanced_features import EnhancedFeatureExtractor
from .xg_history import XGHistory, get_xg_history
from .match_mapper import MatchMapper, map_match

__all__ = [
    # Data processing
    "RawMatchRecord",
    "NormalizedMatch",
    "load_csv",
    "normalize_matches",
    "ingest_directory",
    # API clients
    "SportMonksClient",
    "BetfairClient",
    "WeatherClient",
    "EnhancedFeatureExtractor",
    # New modules
    "XGHistory",
    "get_xg_history",
    "MatchMapper",
    "map_match"
]


