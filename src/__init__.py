"""
STAVKI Value Betting System

Main package for the betting system.
"""

__version__ = "0.1.0"
__author__ = "STAVKI Team"

from .config import Config
from .logging_setup import setup_logging

__all__ = ["Config", "setup_logging", "__version__"]
