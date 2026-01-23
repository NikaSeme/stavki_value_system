"""
EDUCATIONAL DEMONSTRATION: Stake Sizing with Kelly Criterion

Shows optimal bet sizing, bankroll management, and risk control.

⚠️ FOR EDUCATIONAL PURPOSES
Students learn: Kelly criterion, fractional Kelly, bankroll management
"""

import logging
from typing import Dict, List
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StakeSizer:
    """
    Calculate optimal stake sizes using Kelly Criterion.
    
    Kelly Criterion: f = (bp - q) / b
    Where:
    - f = fraction of bankroll to bet
    - b = odds - 1 (net odds)
    - p = probability of winning
    - q = 1 - p
    """
    
    def __init__(
        self,
        kelly_fraction: float = 0.25,
        min_stake: float = 2.0,
        max_stake_pct: float = 0.05  # Max 5% of bankroll per bet
    ):
        """
        Initialize stake sizer.
        
        Args:
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly, conservative)
            min_stake: Minimum bet size
            max_stake_pct: Maximum % of bankroll per bet
        """
        self.kelly_fraction = kelly_fraction
        self.min_stake = min_stake
        self.max_stake_pct = max_stake_pct
        
        logger.info(f"✓ Stake sizer initialized (Kelly fraction: {kelly_fraction})")
    
    def calculate_kelly_stake(
        self,
        bankroll: float,
        win_probability: float,
        odds: float
    ) -> float:
        """
        Calculate optimal stake using Kelly Criterion.
        
        Args:
            bankroll: Total available bankroll
            win_probability: Model's win probability (0-1)
            odds: Decimal odds (e.g., 2.10)
            
        Returns:
            Optimal stake amount
        """
        # Kelly formula
        b = odds - 1  # Net odds
        p = win_probability
        q = 1 - p
        
        # Full Kelly
        kelly = (b * p - q) / b
        
        # Handle negative Kelly (no bet)
        if kelly <= 0:
            logger.warning(f"Negative Kelly ({kelly:.4f}) - no bet recommended")
            return 0.0
        
        # Apply fraction (conservative approach)
        fractional_kelly = kelly * self.kelly_fraction
        
        # Calculate stake
        stake = bankroll * fractional_kelly
        
        # Apply bounds
        max_stake = bankroll * self.max_stake_pct
        stake = max(self.min_stake, min(stake, max_stake))
        
        logger.debug(f"Kelly: {kelly:.4f}, Fractional: {fractional_kelly:.4f}, Stake: £{stake:.2f}")
        
        return stake
    
    def distribute_stake(
        self,
        total_stake: float,
        accounts: List[Dict]
    ) -> Dict[str, float]:
        """
        Distribute stake across multiple accounts.
        
        Args:
            total_stake: Total amount to bet
            accounts: List of available accounts with balances and limits
            
        Returns:
            Dict mapping account_id to stake amount
        """
        distribution = {}
        remaining = total_stake
        
        # Sort by max_stake (descending)
        sorted_accounts = sorted(
            [a for a in accounts if a.get('enabled', True)],
            key=lambda x: x.get('max_stake', 0),
            reverse=True
        )
        
        for account in sorted_accounts:
            if remaining <= 0:
                break
            
            account_id = account['id']
            account_balance = account.get('balance', 0)
            account_max = account.get('max_stake', 100)
            
            # Calculate how much this account can handle
            can_take = min(
                account_max,  # Account limit
                account_balance * 0.10,  # Max 10% of this account's balance
                remaining  # What's left to place
            )
            
            # Only allocate if meets minimum
            if can_take >= self.min_stake:
                distribution[account_id] = round(can_take, 2)
                remaining -= can_take
        
        if remaining > self.min_stake:
            logger.warning(f"Could not fully distribute stake: £{remaining:.2f} remaining")
        
        return distribution
    
    def calculate_ev(self, win_probability: float, odds: float) -> float:
        """
        Calculate Expected Value (EV).
        
        EV = (probability * profit) - (1-probability * stake)
        EV% = ((probability * odds) - 1) * 100
        
        Args:
            win_probability: Model's win probability
            odds: Decimal odds
            
        Returns:
            EV percentage
        """
        ev_pct = ((win_probability * odds) - 1) * 100
        return ev_pct


def test_stake_sizer():
    """Test stake sizing calculations."""
    print("=" * 70)
    print("STAKE SIZER TEST")
    print("=" * 70)
    
    sizer = StakeSizer(kelly_fraction=0.25)
    
    # Test 1: Kelly calculation
    print("\n1. Kelly Criterion Calculation:")
    bankroll = 1000.0
    win_prob = 0.55  # 55% win probability
    odds = 2.10
    
    stake = sizer.calculate_kelly_stake(bankroll, win_prob, odds)
    ev = sizer.calculate_ev(win_prob, odds)
    
    print(f"  Bankroll: £{bankroll:.2f}")
    print(f"  Win Probability: {win_prob:.1%}")
    print(f"  Odds: {odds:.2f}")
    print(f"  EV: {ev:+.2f}%")
    print(f"  Recommended Stake: £{stake:.2f} ({stake/bankroll:.1%} of bankroll)")
    
    # Test 2: Multi-account distribution
    print("\n2. Multi-Account Distribution:")
    total_stake = 150.0
    accounts = [
        {'id': 'account_1', 'balance': 500, 'max_stake': 50, 'enabled': True},
        {'id': 'account_2', 'balance': 1000, 'max_stake': 100, 'enabled': True},
        {'id': 'account_3', 'balance': 300, 'max_stake': 30, 'enabled': True}
    ]
    
    distribution = sizer.distribute_stake(total_stake, accounts)
    
    print(f"  Total Stake: £{total_stake:.2f}")
    print(f"  Distribution:")
    for acct_id, acct_stake in distribution.items():
        print(f"    {acct_id}: £{acct_stake:.2f}")
    print(f"  Total Distributed: £{sum(distribution.values()):.2f}")
    
    # Test 3: Edge cases
    print("\n3. Edge Cases:")
    
    # Negative EV (should return 0)
    negative_stake = sizer.calculate_kelly_stake(1000, 0.40, 2.10)
    print(f"  Negative EV stake: £{negative_stake:.2f} (should be 0)")
    
    # Very high probability
    high_prob_stake = sizer.calculate_kelly_stake(1000, 0.90, 1.50)
    print(f"  High probability stake: £{high_prob_stake:.2f}")
    
    print("\n✅ Stake sizer test complete!")


if __name__ == '__main__':
    test_stake_sizer()
