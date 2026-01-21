"""
Expected Value (EV) calculation for betting.

EV = probability * odds - 1
Positive EV indicates profitable bet in the long run.
"""

from typing import Union

import numpy as np
import pandas as pd

from ..logging_setup import get_logger

logger = get_logger(__name__)


def calculate_ev(
    probability: Union[float, np.ndarray],
    odds: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate Expected Value for a bet.
    
    Formula: EV = p * odds - 1
    
    Args:
        probability: Win probability (0-1)
        odds: Decimal odds (e.g., 2.5 for 3/2)
        
    Returns:
        Expected value (positive = profitable long-term)
        
    Examples:
        >>> calculate_ev(0.5, 2.5)
        0.25  # 25% positive EV
        >>> calculate_ev(0.4, 2.0)
       -0.2  # 20% negative EV
    """
    # Handle edge cases
    if isinstance(probability, (float, int)):
        if probability < 0 or probability > 1:
            logger.warning(f"Invalid probability: {probability}")
            return np.nan
        if odds <= 1.0:
            logger.warning(f"Invalid odds: {odds}")
            return np.nan
    else:
        # Array case
        probability = np.asarray(probability)
        odds = np.asarray(odds)
        
        # Mask invalid values
        invalid_mask = (probability < 0) | (probability > 1) | (odds <= 1.0)
        
        if invalid_mask.any():
            logger.warning(f"{invalid_mask.sum()} invalid probability/odds pairs")
    
    # Calculate EV
    ev = probability * odds - 1.0
    
    return ev


def filter_positive_ev(
    predictions_df: pd.DataFrame,
    ev_threshold: float = 0.0,
    prob_column: str = 'prob_home',
    odds_column: str = 'odds_1'
) -> pd.DataFrame:
    """
    Filter bets with positive EV above threshold.
    
    Args:
        predictions_df: DataFrame with probabilities and odds
        ev_threshold: Minimum EV to include (default: 0.0)
        prob_column: Column name for probability
        odds_column: Column name for odds
        
    Returns:
        Filtered DataFrame with EV column added
    """
    df = predictions_df.copy()
    
    # Calculate EV
    df['ev'] = calculate_ev(df[prob_column], df[odds_column])
    
    # Filter by threshold
    filtered = df[df['ev'] >= ev_threshold].copy()
    
    logger.info(
        f"Filtered {len(filtered)}/{len(df)} bets with EV >= {ev_threshold:.2%}"
    )
    
    return filtered
