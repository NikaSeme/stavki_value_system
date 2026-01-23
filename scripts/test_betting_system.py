"""
Test complete automated betting system.

EDUCATIONAL DEMONSTRATION of automated bet execution.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.account_manager import AccountManager
from src.betting.stake_sizer import StakeSizer
from src.betting.bet_executor import BetExecutor


def test_full_system():
    """Test complete betting automation system."""
    print("=" * 70)
    print("AUTOMATED BETTING SYSTEM - FULL TEST")
    print("=" * 70)
    print("\n⚠️  EDUCATIONAL DEMONSTRATION - DRY RUN MODE\n")
    
    # Initialize components
    print("1. Initializing components...")
    account_mgr = AccountManager(config_path='config/test_accounts.json')
    stake_sizer = StakeSizer(kelly_fraction=0.25)
    executor = BetExecutor(
        account_manager=account_mgr,
        stake_sizer=stake_sizer,
        dry_run=True
    )
    
    # Test bet signal
    signal = {
        'match': 'Manchester City vs Liverpool',
        'market': 'Home Win',
        'odds': 2.15,
        'model_prob': 0.523,
        'ev': 6.7
    }
    
    print("\n2. Bet Signal:")
    print(f"  Match: {signal['match']}")
    print(f"  Market: {signal['market']}")
    print(f"  Odds: {signal['odds']:.2f}")
    print(f"  Model Probability: {signal['model_prob']:.1%}")
    print(f"  Expected Value: +{signal['ev']:.1f}%")
    
    # Execute
    print("\n3. Executing bet...")
    result = executor.execute_bet_signal(signal)
    
    print("\n4. Execution Result:")
    print(f"  ✓ Success: {result['success']}")
    print(f"  ✓ Bets Placed: {result.get('bets_placed', 0)}")
    print(f"  ✓ Total Stake: £{result.get('total_stake', 0):.2f}")
    print(f"  ✓ Accounts Used: {len(result.get('accounts_used', []))}")
    
    if result.get('results'):
        print(f"\n5. Individual Bets:")
        for bet_result in result['results']:
            print(f"  - Account: {bet_result.get('account_id')}")
            print(f"    Stake: £{bet_result.get('stake', 0):.2f}")
            print(f"    Status: {'✓ Placed' if bet_result.get('success') else '✗ Failed'}")
    
    print("\n" + "=" * 70)
    print("✅ FULL SYSTEM TEST COMPLETE")
    print("=" * 70)
    print("\n📚 This demonstrates:")
    print("  - Secure account management (encryption)")
    print("  - Kelly criterion stake sizing")
    print("  - Multi-account distribution")
    print("  - Complete execution pipeline")
    print("  - Transaction logging")
    print("\n⚠️  Remember: Real betting requires understanding")
    print("    legal/ethical implications and platform ToS")


if __name__ == '__main__':
    test_full_system()
