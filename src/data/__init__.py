"""
Data ingestion package for STAVKI betting system.
Handles loading and normalization of football match data.
"""

from .ingestion import ingest_directory, load_csv, normalize_matches
from .schemas import NormalizedMatch, RawMatchRecord

__all__ = [
    "RawMatchRecord",
    "NormalizedMatch",
    "load_csv",
    "normalize_matches",
    "ingest_directory",
]
