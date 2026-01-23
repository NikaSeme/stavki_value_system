"""
EDUCATIONAL DEMONSTRATION: Account Management for Automated Betting

Shows secure credential storage, balance tracking, and limit management.

⚠️ FOR EDUCATIONAL PURPOSES ONLY
Students learn: encryption, credential management, multi-account orchestration
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
from cryptography.fernet import Fernet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BettingAccount:
    """Represents a single betting account."""
    id: str
    type: str  # 'betfair', 'pinnacle', etc.
    username: str
    balance: float
    max_stake: float
    enabled: bool = True
    
    # Encrypted credentials (not stored in dataclass)
    _encrypted_password: Optional[str] = None
    _encrypted_api_key: Optional[str] = None


class AccountManager:
    """
    Manage multiple betting accounts securely.
    
    Features:
    - Encrypted credential storage
    - Balance tracking
    - Limit monitoring
    - Account selection logic
    """
    
    def __init__(self, config_path='config/accounts.json'):
        """Initialize account manager."""
        self.config_path = Path(config_path)
        self.encryption_key = self._get_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        
        self.accounts: List[BettingAccount] = []
        self.load_accounts()
        
        logger.info(f"✓ Account manager initialized ({len(self.accounts)} accounts)")
    
    def _get_encryption_key(self) -> bytes:
        """Get or create encryption key."""
        key_env = 'BETTING_ACCOUNTS_KEY'
        key = os.getenv(key_env)
        
        if not key:
            # Generate new key (for demo)
            key = Fernet.generate_key().decode()
            logger.warning(f"Generated new encryption key. Set {key_env}={key}")
            # In production, save this securely!
        
        return key.encode() if isinstance(key, str) else key
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted: str) -> str:
        """Decrypt sensitive data."""
        return self.cipher.decrypt(encrypted.encode()).decode()
    
    def load_accounts(self):
        """Load accounts from config."""
        if not self.config_path.exists():
            logger.warning(f"No accounts config found: {self.config_path}")
            self._create_demo_config()
            return
        
        with open(self.config_path) as f:
            config = json.load(f)
        
        for acct_data in config.get('accounts', []):
            account = BettingAccount(
                id=acct_data['id'],
                type=acct_data['type'],
                username=acct_data['username'],
                balance=acct_data['balance'],
                max_stake=acct_data['max_stake'],
                enabled=acct_data.get('enabled', True)
            )
            
            # Store encrypted credentials separately
            account._encrypted_password = acct_data.get('password_encrypted')
            account._encrypted_api_key = acct_data.get('api_key_encrypted')
            
            self.accounts.append(account)
        
        logger.info(f"Loaded {len(self.accounts)} accounts")
    
    def get_account(self, account_id: str) -> Optional[BettingAccount]:
        """Get account by ID."""
        for account in self.accounts:
            if account.id == account_id:
                return account
        return None
    
    def get_credentials(self, account_id: str) -> Dict[str, str]:
        """
        Get decrypted credentials for account.
        
        ⚠️ Handle with care - contains sensitive data!
        """
        account = self.get_account(account_id)
        if not account:
            raise ValueError(f"Account not found: {account_id}")
        
        credentials = {'username': account.username}
        
        if account._encrypted_password:
            credentials['password'] = self.decrypt(account._encrypted_password)
        
        if account._encrypted_api_key:
            credentials['api_key'] = self.decrypt(account._encrypted_api_key)
        
        return credentials
    
    def get_available_accounts(self, min_balance: float = 10.0) -> List[BettingAccount]:
        """Get accounts that can place bets."""
        return [
            a for a in self.accounts
            if a.enabled and a.balance >= min_balance
        ]
    
    def update_balance(self, account_id: str, new_balance: float):
        """Update account balance."""
        account = self.get_account(account_id)
        if account:
            old_balance = account.balance
            account.balance = new_balance
            logger.info(f"Balance updated: {account_id} £{old_balance:.2f} → £{new_balance:.2f}")
    
    def deduct_stake(self, account_id: str, stake: float):
        """Deduct stake from account balance."""
        account = self.get_account(account_id)
        if not account:
            raise ValueError(f"Account not found: {account_id}")
        
        if account.balance < stake:
            raise ValueError(f"Insufficient balance: £{account.balance:.2f} < £{stake:.2f}")
        
        account.balance -= stake
        logger.info(f"Stake deducted: {account_id} -£{stake:.2f} (new: £{account.balance:.2f})")
    
    def _create_demo_config(self):
        """Create demo configuration for educational purposes."""
        demo_config = {
            "accounts": [
                {
                    "id": "betfair_demo",
                    "type": "betfair",
                    "username": "demo_user",
                    "password_encrypted": self.encrypt("demo_password"),
                    "api_key_encrypted": self.encrypt("demo_api_key"),
                    "balance": 1000.0,
                    "max_stake": 100.0,
                    "enabled": True
                },
                {
                    "id": "account_2",
                    "type": "generic",
                    "username": "user2",
                    "password_encrypted": self.encrypt("password2"),
                    "balance": 500.0,
                    "max_stake": 50.0,
                    "enabled": True
                }
            ]
        }
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(demo_config, f, indent=2)
        
        logger.info(f"Created demo config: {self.config_path}")
        self.load_accounts()


def test_account_manager():
    """Test account management."""
    print("=" * 70)
    print("ACCOUNT MANAGER TEST")
    print("=" * 70)
    
    manager = AccountManager(config_path='config/test_accounts.json')
    
    print("\n1. Available accounts:")
    for account in manager.get_available_accounts():
        print(f"  - {account.id}: £{account.balance:.2f} (max: £{account.max_stake:.2f})")
    
    print("\n2. Get credentials (encrypted/decrypted):")
    if manager.accounts:
        creds = manager.get_credentials(manager.accounts[0].id)
        print(f"  Username: {creds['username']}")
        print(f"  Password: {'*' * len(creds.get('password', ''))}")
        print(f"  API Key: {'*' * 10}...") if 'api_key' in creds else None
    
    print("\n3. Deduct stake simulation:")
    if manager.accounts:
        account_id = manager.accounts[0].id
        print(f"  Before: £{manager.get_account(account_id).balance:.2f}")
        manager.deduct_stake(account_id, 50.0)
        print(f"  After: £{manager.get_account(account_id).balance:.2f}")
    
    print("\n✅ Account manager test complete!")


if __name__ == '__main__':
    test_account_manager()
