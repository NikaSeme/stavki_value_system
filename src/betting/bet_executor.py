"""
EDUCATIONAL DEMONSTRATION: Automated Bet Execution

Shows complete bet execution pipeline from signal to placement.

⚠️ FOR EDUCATIONAL PURPOSES ONLY
Students learn: API integration, error handling, transaction logging
Real-world use: Only on platforms that allow automation (e.g., Betfair Exchange)
"""

import time
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json

from .account_manager import AccountManager, BettingAccount
from .stake_sizer import StakeSizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BetExecutor:
    """
    Execute bets on betting platforms.
    
    Educational demonstration of:
    - Signal validation
    - Stake calculation
    - Multi-account distribution
    - Error handling
    - Transaction logging
    """
    
    def __init__(
        self,
        account_manager: Optional[AccountManager] = None,
        stake_sizer: Optional[StakeSizer] = None,
        dry_run: bool = True
    ):
        """
        Initialize bet executor.
        
        Args:
            account_manager: Account manager instance
            stake_sizer: Stake sizing instance
            dry_run: If True, simulates bets without placing
        """
        self.account_manager = account_manager or AccountManager()
        self.stake_sizer = stake_sizer or StakeSizer()
        self.dry_run = dry_run
        
        # Execution log
        self.log_file = Path('data/logs/bet_execution.json')
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        mode = "DRY RUN" if dry_run else "LIVE"
        logger.info(f"✓ Bet executor initialized ({mode} mode)")
    
    def execute_bet_signal(self, signal: Dict) -> Dict:
        """
        Execute a bet signal end-to-end.
        
        Workflow:
        1. Validate signal
        2. Calculate stake
        3. Select accounts
        4. Place bets
        5. Handle responses
        6. Log results
        
        Args:
            signal: Bet signal with match, odds, probability, etc.
            
        Returns:
            Execution result dict
        """
        logger.info(f"📊 Processing bet signal: {signal.get('match', 'Unknown')}")
        
        # 1. Validate
        if not self._validate_signal(signal):
            return {'success': False, 'reason': 'Invalid signal'}
        
        # 2. Calculate stake
        total_bankroll = sum(a.balance for a in self.account_manager.get_available_accounts())
        
        stake = self.stake_sizer.calculate_kelly_stake(
            bankroll=total_bankroll,
            win_probability=signal['model_prob'],
            odds=signal['odds']
        )
        
        if stake < self.stake_sizer.min_stake:
            logger.info(f"Stake too small (£{stake:.2f}) - no bet")
            return {'success': False, 'reason': 'Stake below minimum'}
        
        logger.info(f"💰 Calculated stake: £{stake:.2f}")
        
        # 3. Distribute across accounts
        accounts = self.account_manager.get_available_accounts()
        account_dicts = [
            {
                'id': a.id,
                'balance': a.balance,
                'max_stake': a.max_stake,
                'enabled': a.enabled
            }
            for a in accounts
        ]
        
        distribution = self.stake_sizer.distribute_stake(stake, account_dicts)
        
        if not distribution:
            logger.warning("No accounts available for bet placement")
            return {'success': False, 'reason': 'No available accounts'}
        
        logger.info(f"📊 Distribution: {len(distribution)} accounts")
        
        # 4. Execute on each account
        results = []
        for account_id, account_stake in distribution.items():
            result = self._place_bet(
                account_id=account_id,
                signal=signal,
                stake=account_stake
            )
            results.append(result)
        
        # 5. Aggregate results
        success_count = sum(1 for r in results if r.get('success'))
        total_placed = sum(r.get('stake', 0) for r in results if r.get('success'))
        
        execution_result = {
            'success': success_count > 0,
            'bets_placed': success_count,
            'total_stake': total_placed,
            'accounts_used': list(distribution.keys()),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        # 6. Log
        self._log_execution(signal, execution_result)
        
        logger.info(f"✅ Execution complete: {success_count}/{len(distribution)} bets placed")
        
        return execution_result
    
    def _validate_signal(self, signal: Dict) -> bool:
        """Validate bet signal has required fields."""
        required = ['match', 'market', 'odds', 'model_prob']
        
        for field in required:
            if field not in signal:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Odds must be > 1.0
        if signal['odds'] <= 1.0:
            logger.error(f"Invalid odds: {signal['odds']}")
            return False
        
        # Probability must be 0-1
        if not (0 < signal['model_prob'] < 1):
            logger.error(f"Invalid probability: {signal['model_prob']}")
            return False
        
        return True
    
    def _place_bet(
        self,
        account_id: str,
        signal: Dict,
        stake: float
    ) -> Dict:
        """
        Place bet on specific account.
        
        Args:
            account_id: Account to use
            signal: Bet signal
            stake: Amount to bet
            
        Returns:
            Bet placement result
        """
        logger.info(f"🎯 Placing bet on {account_id}: £{stake:.2f} @ {signal['odds']}")
        
        if self.dry_run:
            # Simulate successful placement
            time.sleep(0.1)  # Simulate network call
            
            return {
                'success': True,
                'account_id': account_id,
                'stake': stake,
                'odds': signal['odds'],
                'bet_id': f"dry_run_{int(time.time())}",
                'simulated': True
            }
        
        # LIVE MODE: Actual bet placement
        try:
            account = self.account_manager.get_account(account_id)
            
            if not account:
                return {
                    'success': False,
                    'error': 'Account not found',
                    'account_id': account_id
                }
            
            # Deduct stake from balance
            self.account_manager.deduct_stake(account_id, stake)
            
            # Place bet based on account type
            if account.type == 'betfair':
                result = self._place_betfair_bet(account, signal, stake)
            else:
                # Other account types (demonstration only)
                result = self._place_generic_bet(account, signal, stake)
            
            return result
            
        except Exception as e:
            logger.error(f"Bet placement failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'account_id': account_id
            }
    
    def _place_betfair_bet(
        self,
        account: BettingAccount,
        signal: Dict,
        stake: float
    ) -> Dict:
        """
        Place bet on Betfair Exchange.
        
        ✅ LEGITIMATE - Betfair supports API betting
        """
        try:
            # Import betfair client (would be initialized with account creds)
            # from betfairlightweight import APIClient
            
            logger.info(f"Placing Betfair bet: {signal['match']} {signal['market']}")
            
            # This is where actual Betfair API call would go
            # client = APIClient(username=..., password=..., app_key=...)
            # response = client.betting.place_orders(...)
            
            # For demo, simulate success
            return {
                'success': True,
                'account_id': account.id,
                'stake': stake,
                'odds': signal['odds'],
                'bet_id': f"betfair_{int(time.time())}",
                'platform': 'betfair'
            }
            
        except Exception as e:
            logger.error(f"Betfair bet failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'account_id': account.id
            }
    
    def _place_generic_bet(
        self,
        account: BettingAccount,
        signal: Dict,
        stake: float
    ) -> Dict:
        """
        Generic bet placement (demonstration).
        
        ⚠️ EDUCATIONAL DEMONSTRATION
        Real implementation would depend on specific bookmaker
        """
        logger.info(f"Generic bet placement on {account.type}")
        
        # This would be replaced with actual implementation
        # (e.g., Selenium automation for bookmakers without APIs)
        
        return {
            'success': True,
            'account_id': account.id,
            'stake': stake,
            'odds': signal['odds'],
            'bet_id': f"generic_{int(time.time())}",
            'platform': account.type,
            'demonstration': True
        }
    
    def _log_execution(self, signal: Dict, result: Dict):
        """Log bet execution for audit trail."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'signal': signal,
            'result': result
        }
        
        # Append to log file
        logs = []
        if self.log_file.exists():
            try:
                with open(self.log_file) as f:
                    logs = json.load(f)
            except:
                logs = []
        
        logs.append(log_entry)
        
        # Keep last 1000 executions
        logs = logs[-1000:]
        
        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        logger.debug(f"Logged execution to {self.log_file}")


def test_bet_executor():
    """Test bet execution system."""
    print("=" * 70)
    print("BET EXECUTOR TEST")
    print("=" * 70)
    
    # Initialize in dry-run mode
    executor = BetExecutor(dry_run=True)
    
    # Test signal
    signal = {
        'match': 'Manchester City vs Liverpool',
        'market': 'Home Win',
        'odds': 2.15,
        'model_prob': 0.523,
        'ev': 6.7
    }
    
    print("\n1. Executing bet signal...")
    print(f"  Match: {signal['match']}")
    print(f"  Market: {signal['market']}")
    print(f"  Odds: {signal['odds']:.2f}")
    print(f"  Model Prob: {signal['model_prob']:.1%}")
    print(f"  EV: +{signal['ev']:.1f}%")
    
    result = executor.execute_bet_signal(signal)
    
    print(f"\n2. Execution Result:")
    print(f"  Success: {result['success']}")
    print(f"  Bets Placed: {result.get('bets_placed', 0)}")
    print(f"  Total Stake: £{result.get('total_stake', 0):.2f}")
    print(f"  Accounts: {len(result.get('accounts_used', []))}")
    
    print("\n✅ Bet executor test complete!")


if __name__ == '__main__':
    test_bet_executor()
