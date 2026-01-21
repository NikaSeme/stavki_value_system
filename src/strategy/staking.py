"""
Kelly Criterion staking strategy.

Kelly stake = (p * odds - 1) / (odds - 1) * bankroll
With fractional Kelly and maximum stake caps for risk management.
"""

from typing import Union

import numpy as np

from ..logging_setup import get_logger

logger = get_logger(__name__)


def kelly_stake(
    probability: Union[float, np.ndarray],
    odds: Union[float, np.ndarray],
    bankroll: float = 1000.0,
    kelly_fraction: float = 1.0,
    max_stake_fraction: float = 0.05
) -> Union[float, np.ndarray]:
    """
    Calculate Kelly Criterion stake size.
    
    Formula:
        f = (p * odds - 1) / (odds - 1)
        stake = f * kelly_fraction * bankroll
        capped at max_stake_fraction * bankroll
    
    Args:
        probability: Win probability (0-1)
        odds: Decimal odds
        bankroll: Total bankroll
        kelly_fraction: Kelly fraction (0.5 = half Kelly, safer)
        max_stake_fraction: Maximum stake as fraction of bankroll
        
    Returns:
        Stake size (non-negative, capped)
        
    Examples:
        >>> kelly_stake(0.55, 2.0, 1000, 0.5, 0.05)
        50.0  # 5% of bankroll (capped)
        >>> kelly_stake(0.45, 2.0, 1000, 0.5, 0.05)
        0.0  # Negative EV, no bet
    """
    # Convert to arrays for vectorization
    is_scalar = isinstance(probability, (float, int))
    
    if is_scalar:
        probability = np.array([probability])
        odds = np.array([odds])
    else:
        probability = np.asarray(probability)
        odds = np.asarray(odds)
    
    # Initialize stakes
    stakes = np.zeros_like(probability, dtype=float)
    
    # Edge cases
    invalid_mask = (
        (probability <= 0) | 
        (probability >= 1) | 
        (odds <= 1.0) |
        np.isnan(probability) |
        np.isnan(odds)
    )
    
    if invalid_mask.any():
        logger.debug(f"{invalid_mask.sum()} invalid probability/odds pairs")
    
    # Valid bets
    valid_mask = ~invalid_mask
    
    if valid_mask.any():
        p = probability[valid_mask]
        o = odds[valid_mask]
        
        # Kelly formula: f = (p * o - 1) / (o - 1)
        kelly_fraction_optimal = (p * o - 1.0) / (o - 1.0)
        
        # Apply fractional Kelly
        kelly_fraction_adjusted = kelly_fraction_optimal * kelly_fraction
        
        # Calculate stake
        stake_amount = kelly_fraction_adjusted * bankroll
        
        # Ensure non-negative
        stake_amount = np.maximum(stake_amount, 0.0)
        
        # Cap at maximum stake
        max_stake = max_stake_fraction * bankroll
        stake_amount = np.minimum(stake_amount, max_stake)
        
        stakes[valid_mask] = stake_amount
    
    # Return scalar if input was scalar
    if is_scalar:
        return float(stakes[0])
    
    return stakes


def fractional_kelly(
    probability: Union[float, np.ndarray],
    odds: Union[float, np.ndarray],
    bankroll: float = 1000.0,
    fraction: float = 0.5,
    max_stake_pct: float = 5.0
) -> Union[float, np.ndarray]:
    """
    Fractional Kelly with percentage-based max stake.
    
    Convenience wrapper for kelly_stake with percentage max stake.
    
    Args:
        probability: Win probability
        odds: Decimal odds
        bankroll: Total bankroll
        fraction: Kelly fraction (default: 0.5 = half Kelly)
        max_stake_pct: Maximum stake percentage (default: 5%)
        
    Returns:
        Stake size
    """
    return kelly_stake(
        probability,
        odds,
        bankroll,
        kelly_fraction=fraction,
        max_stake_fraction=max_stake_pct / 100.0
    )
