# STAVKI Value Betting System

Advanced value betting system using machine learning, ensemble methods, and Kelly criterion staking.

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-repo/stavki.git
cd stavki_value_system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
# Edit .env and add your API keys
```

### Configuration

Edit `.env` and add your API keys:

```bash
# The Odds API (get free key at https://the-odds-api.com)
ODDS_API_KEY=your_api_key_here

# Telegram Bot (optional, for notifications)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_ALLOWED_USERS=your_user_id_here
```

## 📊 Odds API Pipeline

Fetch and normalize live odds from The Odds API:

```bash
# Activate venv first
source venv/bin/activate

# Fetch EPL odds
python3 run_odds_pipeline.py --sport soccer_epl --regions eu --markets h2h

# Fetch NBA odds
python3 run_odds_pipeline.py --sport basketball_nba --regions us

# Custom output directory
python3 run_odds_pipeline.py --sport americanfootball_nfl --output-dir my_outputs/

# See all options
python3 run_odds_pipeline.py --help
```

**Outputs:**
- `outputs/odds/raw_{sport}_{timestamp}.json` - Raw API response
- `outputs/odds/normalized_{sport}_{timestamp}.csv` - Normalized odds data

## 💎 Live Value Finder

Find value bets by comparing model probabilities with best available odds:

```bash
# Activate venv first
source venv/bin/activate

# Find value bets from latest odds
python run_value_finder.py --sport soccer_epl --ev-threshold 0.05

# Show top 5 bets with Telegram alert
python run_value_finder.py --sport soccer_epl --top-n 5 --telegram

# Custom directories
python run_value_finder.py --odds-dir my_odds/ --output-dir my_value/

# See all options
python run_value_finder.py --help
```

**Outputs:**
- `outputs/value/value_{sport}_{timestamp}.csv` - Ranked value bets
- `outputs/value/value_{sport}_{timestamp}.json` - Detailed bet information
- Optional Telegram alert with top picks

**How it works:**
1. Loads latest normalized odds for the sport
2. Selects best price across all bookmakers for each outcome
3. Computes no-vig probabilities to remove bookmaker margin
4. Gets model predictions (currently using simple baseline model)
5. Calculates EV = p_model × odds - 1
6. Ranks bets by expected value and saves results

## 🎯 Betting Pipeline

Run complete betting analysis:

```bash
# Run with test data
python -m src.cli run \
  --matches data/processed/features.csv \
  --odds data/processed/odds.csv \
  --bankroll 1000 \
  --ev-threshold 0.10 \
  --max-bets 5 \
  --output outputs/bets

# See help
python -m src.cli run --help
```

## 🤖 Telegram Bot

Start the Telegram bot for notifications:

```bash
# Configure .env with TELEGRAM_BOT_TOKEN and TELEGRAM_ALLOWED_USERS first

# Run bot
python scripts/run_bot.py
```

**Bot Commands:**
- `/start` - Welcome and command list
- `/run` - Run betting pipeline
- `/run 1000 0.15` - Custom bankroll and EV threshold
- `/latest` - View latest recommendations
- `/status` - System status
- `/stats` - Performance statistics

## 📈 Evaluation

Evaluate betting performance:

```bash
# Evaluate from results CSV
python -m src.cli eval \
  --results data/results.csv \
  --output outputs/evaluation
```

## 🧪 Testing

Run tests:

```bash
# All tests
pytest

# With coverage
pytest --cov=src tests/

# Specific test file
pytest tests/test_kelly.py -v
```

## 📁 Project Structure

```
stavki_value_system/
├── src/
│   ├── bot/              # Telegram bot
│   ├── config/           # Configuration and env loading
│   ├── data/             # Data ingestion and odds API
│   ├── features/         # Feature engineering
│   ├── models/           # ML models (Poisson, ML, Ensemble)
│   ├── pipeline/         # End-to-end pipelines
│   └── strategy/         # EV calculation and staking
├── tests/                # Test suite
├── data/                 # Data files
├── outputs/              # Pipeline outputs
├── run_odds_pipeline.py  # Odds fetching entrypoint
└── scripts/              # Utility scripts
```

## 🔑 API Keys

### The Odds API

1. Visit [the-odds-api.com](https://the-odds-api.com)
2. Sign up for free account
3. Get API key (500 requests/month free)
4. Add to `.env`: `ODDS_API_KEY=your_key_here`

### Telegram Bot

1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Create new bot with `/newbot`
3. Get your user ID from [@userinfobot](https://t.me/userinfobot)
4. Add to `.env`:
   ```
   TELEGRAM_BOT_TOKEN=your_token_here
   TELEGRAM_ALLOWED_USERS=your_user_id_here
   ```

## 📝 Available Sports

Common sport keys for odds API:

- **Soccer**: `soccer_epl`, `soccer_spain_la_liga`, `soccer_germany_bundesliga`
- **Basketball**: `basketball_nba`, `basketball_euroleague`
- **American Football**: `americanfootball_nfl`, `americanfootball_ncaaf`
- **Baseball**: `baseball_mlb`
- **Ice Hockey**: `icehockey_nhl`

See full list: `python3 run_odds_pipeline.py --sport list`

## ⚠️ Important Notes

- **Never commit `.env`** - It contains secrets
- **API Rate Limits** - Free tier: 500 requests/month
- **Cost per request** - markets × regions (keep tight!)
- **Dry run mode** - Set `DRY_RUN=true` in `.env` for testing

## 📊 Example Workflow

```bash
# 1. Fetch latest odds
python run_odds_pipeline.py --sport soccer_epl

# 2. Find value bets
python run_value_finder.py --sport soccer_epl --ev-threshold 0.05 --telegram

# 3. Review recommendations
cat outputs/value/value_soccer_epl_*.csv

# Alternative: Run full betting pipeline with features
python -m src.cli run \
  --matches outputs/odds/normalized_soccer_epl_latest.csv \
  --odds outputs/odds/normalized_soccer_epl_latest.csv \
  --bankroll 1000 \
  --ev-threshold 0.10

# 4. Track results and evaluate
python -m src.cli eval --results data/results.csv
```

## 🤝 Contributing

1. Run tests: `pytest`
2. Format code: `black src/ tests/`
3. Type check: `mypy src/`
4. Commit and push

## 📄 License

MIT License - See LICENSE file

## 🆘 Troubleshooting

### ModuleNotFoundError

```bash
# Ensure venv is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### ODDS_API_KEY not found

```bash
# Check .env file exists
ls -la .env

# Verify key is set
grep ODDS_API_KEY .env

# Try api1.env as fallback
cat api1.env
```

### Import errors

```bash
# Run as module from project root
cd stavki_value_system
python -m src.cli --help
```

---

**Built with:** Python, scikit-learn, pandas, Click, python-telegram-bot, The Odds API
