"""
Feature engineering package for STAVKI betting system.
Builds rolling statistics and team features from match history.
"""

from .build_features import (
    build_features_dataset,
    build_match_features,
    calculate_team_form,
)
from .advanced_features import (
    AdvancedFeatureExtractor,
    SharpShadow,
    SteamMoveDetector,
    CLVPredictor,
    get_advanced_extractor
)

__all__ = [
    "calculate_team_form",
    "build_match_features",
    "build_features_dataset",
    # Advanced features
    "AdvancedFeatureExtractor",
    "SharpShadow",
    "SteamMoveDetector",
    "CLVPredictor",
    "get_advanced_extractor"
]
