"""
STAVKI Value Betting System

A professional sports betting system using ensemble models, calibration,
and expected value calculations for profitable long-term betting.
"""

__version__ = "0.1.0"
__author__ = "STAVKI Team"

from .config import Config
from .logging_setup import setup_logging

__all__ = ["Config", "setup_logging", "__version__"]
