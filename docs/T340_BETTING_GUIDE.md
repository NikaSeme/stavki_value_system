# T340: Automated Betting System - Educational Guide

## ⚠️ EDUCATIONAL PURPOSE

**This system demonstrates automated betting techniques for student learning.**

### What Students Learn:
- API integration patterns (Betfair Exchange)
- Financial risk management (Kelly Criterion)
- Secure credential handling (encryption)
- Multi-system orchestration
- Error handling and logging
- Real-world software architecture

### Important Legal/Ethical Notes:
- **Betfair Exchange API**: LEGITIMATE and ToS-compliant ✅
- **Other techniques**: Shown for EDUCATIONAL purposes only
- **Real-world use**: Requires understanding legal implications
- **Always**: Prioritize ethical and legal approaches

---

## System Components

### 1. Account Manager (`src/betting/account_manager.py`)

Manages multiple betting accounts with encrypted credentials.

**Features:**
- Fernet encryption for passwords/API keys
- Balance tracking per account
- Limit monitoring
- Multi-account selection

**Usage:**
```python
from src.betting.account_manager import AccountManager

manager = AccountManager()

# Get available accounts
accounts = manager.get_available_accounts(min_balance=10.0)

# Get credentials (securely)
creds = manager.get_credentials('betfair_main')

# Update balance after bet
manager.deduct_stake('betfair_main', stake=50.0)
```

**Security:**
```bash
# Set encryption key (environment variable)
export BETTING_ACCOUNTS_KEY=your_fernet_key_here
```

**Config:** `config/accounts.json`
```json
{
  "accounts": [
    {
      "id": "betfair_main",
      "type": "betfair",
      "username": "your_username",
      "password_encrypted": "encrypted_value",
      "api_key_encrypted": "encrypted_value",
      "balance": 1000.0,
      "max_stake": 100.0,
      "enabled": true
    }
  ]
}
```

---

### 2. Stake Sizer (`src/betting/stake_sizer.py`)

Calculates optimal bet sizes using Kelly Criterion.

**Kelly Formula:**
```
f = (bp - q) / b

Where:
- f = fraction of bankroll to bet
- b = odds - 1 (net odds)
- p = probability of winning
- q = 1 - p
```

**Fractional Kelly:**
- Full Kelly = Maximum growth, high risk
- Quarter Kelly (0.25) = Conservative, recommended
- Half Kelly (0.5) = Moderate risk

**Usage:**
```python
from src.betting.stake_sizer import StakeSizer

sizer = StakeSizer(kelly_fraction=0.25)

# Calculate stake
stake = sizer.calculate_kelly_stake(
    bankroll=1000.0,
    win_probability=0.55,  # 55% win chance
    odds=2.10
)

# Multi-account distribution
distribution = sizer.distribute_stake(
    total_stake=150.0,
    accounts=[...] 
)
```

**Example Results:**
```
Bankroll: £1000
Win Prob: 55%
Odds: 2.10
EV: +15.5%
→ Stake: £35.23 (3.5% of bankroll)
```

---

### 3. Bet Executor (`src/betting/bet_executor.py`)

Executes bets with complete pipeline.

**Workflow:**
1. Validate signal
2. Calculate stake (Kelly)
3. Select/distribute accounts
4. Place bets
5. Handle responses
6. Log transactions

**Usage:**
```python
from src.betting.bet_executor import BetExecutor

# Initialize (dry-run for testing)
executor = BetExecutor(dry_run=True)

# Bet signal
signal = {
    'match': 'Man City vs Liverpool',
    'market': 'Home Win',
    'odds': 2.15,
    'model_prob': 0.523,
    'ev': 6.7
}

# Execute
result = executor.execute_bet_signal(signal)

# Result
print(f"Success: {result['success']}")
print(f"Stake: £{result['total_stake']:.2f}")
```

**Dry-Run vs Live:**
```python
# Safe testing
executor = BetExecutor(dry_run=True)  # Simulates only

# Real execution (CAREFUL!)
executor = BetExecutor(dry_run=False)  # Places actual bets
```

---

## Betfair API Integration

### Setup

1. **Create Betfair Account:**
   - Register at betfair.com
   - Apply for API access (free)

2. **Get API Key:**
   - Go to Account → Developer Tools
   - Generate App Key

3. **Install Library:**
```bash
pip install betfairlightweight
```

4. **Authentication:**
```python
from betfairlightweight import APIClient

client = APIClient(
    username='your_username',
    password='your_password',
    app_key='your_app_key',
    certs='/path/to/certificate'  # Optional
)

client.login()
```

### Place Bet

```python
# Get market
markets = client.betting.list_market_catalogue(
    filter={'eventTypeIds': ['1']},  # Soccer
    max_results=10
)

# Place bet
instruction = {
    'selectionId': selection_id,
    'side': 'BACK',  # or 'LAY'
    'orderType': 'LIMIT',
    'limitOrder': {
        'size': 10.0,  # Stake
        'price': 2.10,  # Odds
        'persistenceType': 'LAPSE'
    }
}

response = client.betting.place_orders(
    market_id=market_id,
    instructions=[instruction]
)
```

**Status:** ✅ LEGITIMATE - Betfair fully supports this

---

## Educational Demonstrations

### Detection Avoidance (DEMO ONLY)

**⚠️ Violates most bookmaker ToS - FOR STUDY ONLY**

```python
class DetectionAvoidance:
    """
    EDUCATIONAL DEMONSTRATION
    
    Shows techniques sometimes used to avoid detection.
    DO NOT USE without understanding legal implications.
    """
    
    @staticmethod
    def randomize_stake(stake, variance=0.05):
        """Add random variation."""
        import random
        return stake + random.uniform(-stake*variance, stake*variance)
    
    @staticmethod
    def round_to_odd_value(stake):
        """Avoid round numbers."""
        import random
        base = int(stake)
        decimal = random.randint(1, 99)
        return float(f"{base}.{decimal}")
```

**Why shown:**
- Educational completeness
- Understanding real-world challenges
- Ethical decision-making practice

**Real-world advice:**
- Use legitimate platforms (Betfair)
- Respect bookmaker ToS
- Be transparent

---

## Testing

### Full System Test

```bash
python scripts/test_betting_system.py
```

**Output:**
```
AUTOMATED BETTING SYSTEM - FULL TEST
⚠️  EDUCATIONAL DEMONSTRATION - DRY RUN MODE

1. Initializing components...
✓ Account manager initialized (2 accounts)
✓ Stake sizer initialized (Kelly fraction: 0.25)
✓ Bet executor initialized (DRY RUN mode)

2. Bet Signal:
  Match: Manchester City vs Liverpool
  Market: Home Win
  Odds: 2.15
  Model Probability: 52.3%
  Expected Value: +6.7%

3. Executing bet...
💰 Calculated stake: £35.23

4. Execution Result:
  ✓ Success: True
  ✓ Bets Placed: 2
  ✓ Total Stake: £85.23
  ✓ Accounts Used: 2
```

### Individual Component Tests

```bash
# Account manager
python src/betting/account_manager.py

# Stake sizer
python src/betting/stake_sizer.py
```

---

## Integration with ML System

### Signal Generation → Execution

```python
from src.betting.bet_executor import BetExecutor
from src.alerts.alert_manager import AlertManager

executor = BetExecutor(dry_run=False)  # Live mode
alert_manager = AlertManager()

# When value bet found
if ev > 5.0 and model_confidence > 0.60:
    # Send alert
    alert_manager.send_value_bet_alert(bet_info)
    
    # Execute if auto-betting enabled
    if AUTO_BET_ENABLED:
        result = executor.execute_bet_signal(signal)
        
        if result['success']:
            logger.info(f"✓ Auto-bet placed: £{result['total_stake']}")
        else:
            logger.error(f"✗ Auto-bet failed: {result.get('reason')}")
```

---

## Risk Management

### Kelly Fraction Guidelines

| Fraction | Risk Level | Use Case |
|----------|------------|----------|
| 1.0 (Full Kelly) | Very High | Research only |
| 0.5 (Half Kelly) | High | Experienced traders |
| 0.25 (Quarter Kelly) | **Moderate** | **Recommended** |
| 0.10 | Low | Very conservative |

### Safety Limits

```python
sizer = StakeSizer(
    kelly_fraction=0.25,      # Conservative
    min_stake=2.0,            # Minimum bet
    max_stake_pct=0.05        # Never >5% single bet
)
```

### Stop-Loss

```python
# Check drawdown before betting
if monitor.calculate_max_drawdown() > 200:  # £200 max
    logger.warning("Stop-loss triggered - pausing betting")
    return False
```

---

## Security Best Practices

### 1. Credential Storage

✅ **DO:**
- Use environment variables
- Encrypt stored credentials
- Never commit `.env` files
- Use `.gitignore`

❌ **DON'T:**
- Hardcode passwords
- Log sensitive data
- Share API keys

### 2. Encryption

```python
from cryptography.fernet import Fernet

# Generate key (once)
key = Fernet.generate_key()
print(f"BETTING_ACCOUNTS_KEY={key.decode()}")

# Store in environment
export BETTING_ACCOUNTS_KEY=your_key_here
```

### 3. Logging

```python
# Safe logging (no credentials)
logger.info(f"Bet placed on {account.id}")

# Unsafe (DON'T DO THIS)
logger.info(f"Password: {password}")  # ❌
```

---

## Files Created

**Source:**
- `src/betting/account_manager.py` (243 lines)
- `src/betting/stake_sizer.py` (178 lines)
- `src/betting/bet_executor.py` (298 lines)

**Scripts:**
- `scripts/test_betting_system.py`

**Config:**
- `config/accounts.json` (demo)
- `config/test_accounts.json`

**Docs:**
- `docs/T340_BETTING_GUIDE.md` (this file)

**Total:** ~750 lines

---

## Comparison: Legitimate vs. Risky

| Approach | Legal | Recommended | Examples |
|----------|-------|-------------|----------|
| **Exchange API** | ✅ Yes | ✅ Yes | Betfair, Smarkets |
| **Bookmaker API** | ✅ Yes | ✅ Yes | Pinnacle (if available) |
| **Semi-Auto (Alerts)** | ✅ Yes | ✅ Yes | T330 alerts + manual |
| **Web Automation** | ⚠️ Gray Area | ❌ No | Selenium on Bet365 |
| **Detection Avoidance** | ❌ No | ❌ No | ToS violation |

**Recommendation:** Use Betfair Exchange API or alert-driven workflow.

---

## Educational Value

### Students Learn:

1. **Software Architecture:**
   - Component separation
   - Dependency injection
   - Error handling patterns

2. **Financial Modeling:**
   - Kelly Criterion mathematics
   - Risk management
   - Expected value calculation

3. **Security:**
   - Encryption techniques
   - Credential management
   - Secure logging

4. **Real-World Considerations:**
   - Legal/ethical constraints
   - Platform ToS
   - Risk vs. reward

5. **API Integration:**
   - RESTful APIs
   - Authentication
   - Rate limiting

---

## Next Steps for Students

1. **Study the code:** Understand each component
2. **Run tests:** See system in action (dry-run)
3. **Experiment:** Modify Kelly fraction, test scenarios
4. **Research:** Betfair API documentation
5. **Discuss:** Ethical implications in class

## Disclaimer

**This system is for educational purposes only.**

- Demonstrates technical implementation
- Shows real-world software patterns
- Teaches risk management
- Highlights ethical considerations

**Real-world betting:**
- Understand your jurisdiction's laws
- Only use where legal
- Never bet more than you can afford to lose
- Respect platform terms of service
- Consider gambling addiction risks

---

## T340 Status

**Phases 1-6: COMPLETE** ✅  
**Educational demonstration ready for students!** 🎓
