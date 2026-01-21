"""
Feature engineering package for STAVKI betting system.
Builds rolling statistics and team features from match history.
"""

from .build_features import (
    build_features_dataset,
    build_match_features,
    calculate_team_form,
)

__all__ = [
    "calculate_team_form",
    "build_match_features",
    "build_features_dataset",
]
