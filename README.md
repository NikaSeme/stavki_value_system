# STAVKI Value Betting System

> ⚠️ **DISCLAIMER**: This is an educational project. Sports betting involves financial risk. Always bet responsibly and never bet more than you can afford to lose. By default, the system runs in **DRY_RUN** mode to prevent accidental real betting.

## Overview

STAVKI is a professional sports betting system that uses:
- **Ensemble Models**: Statistical (Poisson/Dixon-Coles), ML (XGBoost/LightGBM), and Neural Networks
- **Probability Calibration**: Platt scaling and isotonic regression for accurate predictions
- **Expected Value (EV)**: Mathematical edge calculation and value bet filtering
- **Risk Management**: Kelly criterion with fractional betting and bankroll protection
- **Multi-bookmaker**: Support for multiple betting platforms with limit avoidance strategies

## Architecture

```
┌─────────────────┐
│  Data Sources   │  ← Historical results, xG, Elo ratings, odds, social signals
└────────┬────────┘
         │
┌────────▼────────┐
│Feature Engineering│
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──┐ ┌───────┐
│Model A│ │Model│ │Model C│
│(Stats)│ │  B  │ │(Neural)│
│Poisson│ │ ML  │ │ LSTM  │
└───┬───┘ └──┬──┘ └───┬───┘
    │        │        │
    └────┬───┴───┬────┘
         │       │
    ┌────▼───────▼────┐
    │  Meta-Ensemble  │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  Calibration    │ ← Platt/Isotonic
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  EV Filtering   │ ← Min threshold 8%
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ Staking (Kelly) │ ← Risk management
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ Execution/Alert │ ← Optional auto-betting
    └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.11 or higher
- macOS or Ubuntu (no GPU required for MVP)

### Setup

1. **Clone or navigate to the project directory**:
   ```bash
   cd stavki_value_system
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings (API keys, parameters, etc.)
   ```

5. **Verify installation**:
   ```bash
   python -m src.cli --help
   python -m src.cli check
   ```

## Quick Start

### 1. Configuration Check

Verify your configuration is valid:

```bash
python -m src.cli config-show
python -m src.cli config-validate
```

### 2. Run System Check

Check that all directories and logging are working:

```bash
python -m src.cli check
```

Expected output:
```
✓ Configuration valid
✓ Data directory exists: data/
✓ Models directory exists: models/
✓ Outputs directory exists: outputs/
✓ Logs directory created: logs/
```

### 3. Analyze Matches (Placeholder)

```bash
python -m src.cli analyze
```

*Note: Full analysis requires implementation of data ingestion and models (future tasks)*

### 4. Run Backtest (Placeholder)

```bash
python -m src.cli backtest --start-date 2024-01-01 --end-date 2024-12-31
```

### 5. Live Monitoring (Placeholder)

```bash
python -m src.cli monitor
```

## Configuration

All configuration is done via environment variables in `.env` file:

### Essential Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DRY_RUN` | `true` | **IMPORTANT**: Set to `false` only to enable real betting |
| `LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `MIN_EV_THRESHOLD` | `0.08` | Minimum 8% edge required for value bet |
| `KELLY_FRACTION` | `0.25` | Conservative Kelly (25% of full Kelly) |
| `MAX_STAKE_PERCENT` | `5.0` | Maximum 5% of bankroll per bet |
| `INITIAL_BANKROLL` | `1000.0` | Starting bankroll |

### API Keys (Optional)

Add these to `.env` when ready to connect to data sources:

```bash
BETFAIR_API_KEY=your_key_here
PINNACLE_API_KEY=your_key_here
ODDS_API_KEY=your_key_here
TWITTER_BEARER_TOKEN=your_token_here
TELEGRAM_BOT_TOKEN=your_token_here
```

## Project Structure

```
stavki_value_system/
├── src/                    # Main source package
│   ├── __init__.py
│   ├── config.py          # Type-safe configuration
│   ├── logging_setup.py   # Structured logging
│   └── cli.py             # Command-line interface
├── tests/                 # Unit tests
│   ├── __init__.py
│   └── test_config.py
├── data/                  # Data directory (created on first run)
├── models/                # Trained models (created on first run)
├── outputs/               # Results and reports
├── logs/                  # Application logs
├── .env.example           # Example environment variables
├── .gitignore
├── requirements.txt       # Python dependencies
└── README.md
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_config.py -v
```

### Code Quality

```bash
# Type checking
mypy src/

# Code formatting
black src/ tests/

# Linting
flake8 src/ tests/
```

## Safety Features

### 🔒 DRY_RUN Mode (Default)

By default, the system runs in **DRY_RUN** mode:
- No real bets are placed
- All operations are simulated and logged
- Perfect for testing and development

To enable real betting (⚠️ **USE WITH CAUTION**):
```bash
# In .env file
DRY_RUN=false
```

### 🛡️ Risk Protections

- **Kelly Fraction**: Conservative 25% Kelly by default (reduces variance)
- **Max Stake**: Never risk more than 5% of bankroll on single bet
- **Max Daily Loss**: Stop if daily losses exceed 10%
- **Max Drawdown**: Alert if total drawdown exceeds 20%

## Roadmap

This is **T001_bootstrap** - the initial scaffold. Future tasks:

- [ ] **T010**: Data ingestion (match results, odds, xG data)
- [ ] **T020**: Feature engineering pipeline
- [ ] **T030**: Model A - Statistical (Poisson, Dixon-Coles, Elo)
- [ ] **T040**: Model B - ML (XGBoost, LightGBM)
- [ ] **T050**: Ensemble & calibration (meta-model, Platt scaling)
- [ ] **T060**: EV calculation & staking logic
- [ ] **T070**: Backtesting framework
- [ ] **T080**: Notifications (Telegram alerts)
- [ ] **T090**: Optional execution module

## Resources

Based on research and best practices from:
- Dixon-Coles (1997) - Poisson model for football
- Hubáček et al. (2019) - ML for sports betting
- Pinnacle's closing line efficiency research
- RebelBetting blog - Bookmaker limit avoidance
- Various academic papers on calibration and risk management

## License

Educational use only. No warranty provided.

## Support

For issues or questions:
1. Check logs in `logs/app.log`
2. Run `python -m src.cli check` for diagnostics
3. Review configuration with `python -m src.cli config-show`

---

**Remember**: Sports betting should be treated as an investment with proper risk management, not gambling. The math works only on long-term expected value with sufficient bankroll and discipline.
