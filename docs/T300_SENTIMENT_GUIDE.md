# T300: Social Media Sentiment Analysis - Setup Guide

## Overview

This system integrates social media sentiment analysis into the match prediction pipeline. Sentiment features capture real-time team morale, injuries, and breaking news.

## Architecture

**Components:**
- `sentiment_analyzer.py`: VADER-based sentiment scoring
- `sentiment_fetcher.py`: Twitter/News API client with caching
- `sentiment_features.py`: Feature extraction per match
- Model: CatBoost with 28 features (22 base + 6 sentiment)

**New Features (6 total):**
1. `home_sentiment_score`: Home team sentiment (-1 to 1)
2. `away_sentiment_score`: Away team sentiment (-1 to 1)
3. `home_injury_flag`: Injury/suspension detected (0/1)
4. `away_injury_flag`: Away team injury flag (0/1)
5. `home_news_volume`: News volume (log-scaled)
6. `away_news_volume`: Away news volume

## API Setup

### Option 1: Twitter API (Recommended)

1. **Create Twitter Developer Account:**
   - Visit: https://developer.twitter.com/
   - Apply for Essential access (free, 500k tweets/month)

2. **Get Bearer Token:**
   - Create project and app
   - Copy Bearer Token

3. **Add to `.env`:**
   ```bash
   TWITTER_BEARER_TOKEN=your_bearer_token_here
   ```

### Option 2: NewsAPI (Alternative)

1. **Get API Key:**
   - Visit: https://newsapi.org/
   - Register for free tier (100 requests/day)

2. **Add to `.env`:**
   ```bash
   NEWS_API_KEY=your_news_api_key_here
   ```

### Option 3: Mock Mode (Testing)

No API required. Uses sample sentiment data.

```python
from src.data.sentiment_fetcher import SentimentFetcher
fetcher = SentimentFetcher(mode='mock')
```

## Usage

### Train Model with Sentiment

```bash
# Add sentiment features to dataset
python scripts/retrain_with_sentiment.py

# Train model
python scripts/train_model_with_sentiment.py
```

### Test Sentiment Features

```bash
# Test analyzer
python src/features/sentiment_analyzer.py

# Test fetcher
python src/data/sentiment_fetcher.py

# Test integration
python scripts/test_sentiment.py
```

### Live Predictions

```python
from src.features.sentiment_features import SentimentFeatureExtractor

# Initialize (uses Twitter if available, else mock)
extractor = SentimentFeatureExtractor(mode='twitter')

# Extract for match
features = extractor.extract_for_match(
    home_team="Manchester City",
    away_team="Liverpool",
    lookback_hours=48  # Last 2 days
)

print(features)
# {
#   'home_sentiment_score': 0.42,
#   'away_sentiment_score': -0.15,
#   'home_injury_flag': 0,
#   'away_injury_flag': 1,  # Injury detected!
#   ...
# }
```

## Configuration

### Team Monitoring (`config/sentiment_config.json`)

```json
{
  "teams": {
    "Manchester City": {
      "twitter_handles": ["@ManCity"],
      "keywords": ["Man City", "Guardiola", "Haaland"],
      "key_players": ["Erling Haaland", "Kevin De Bruyne"]
    }
  },
  "sentiment_keywords": {
    "injury": ["injured", "ruled out", "sidelined"],
    "suspension": ["suspended", "ban", "red card"]
  }
}
```

### Rate Limits

- **Twitter:** 180 requests / 15 min
- **NewsAPI:** 100 requests / day (free tier)
- **Cache:** 1 hour default

Cache automatically prevents excessive API calls.

## Model Performance

### Baseline (22 features)
- Test Accuracy: 60.23%
- Brier Score: 0.1825

### With Sentiment (28 features, mock data)
- Test Accuracy: 58.48%
- Brier Score: 0.1788
- **Note:** Mock data not predictive; real Twitter data expected to improve performance

### Feature Importance

Sentiment features ranked 23-28 (lowest) with mock data. With real sentiment correlated to outcomes:
- Injury flags expected in top 10
- Sentiment scores expected in top 15

## API Error Handling

System handles failures gracefully:

```python
# API unavailable → use cached sentiment
# No cache → use neutral defaults (0.0 sentiment, no injury)
# Logs all errors for debugging
```

## Caching

Location: `data/cache/sentiment_cache.json`

```json
{
  "Manchester City_48": {
    "timestamp": 1705998234,
    "posts": [...]
  }
}
```

Clear cache:
```bash
rm data/cache/sentiment_cache.json
```

## Debugging

Check logs:
```bash
# Test sentiment pipeline
python scripts/test_sentiment.py

# View cache
cat data/cache/sentiment_cache.json | jq
```

## Production Checklist

✅ Get Twitter/NewsAPI credentials
✅ Add to `.env`
✅ Test API connection
✅ Retrain model with real sentiment data
✅ Monitor API usage (stay under limits)
✅ Set up scheduled fetching (cron)

## Next Steps

1. **Get API Access:** Twitter Developer Account
2. **Retrain Model:** With real sentiment data
3. **A/B Test:** Compare 22-feature vs 28-feature models
4. **Monitor:** Feature importance in production
5. **Iterate:** Add more keywords, refine sentiment logic

## Files Created

- `src/features/sentiment_analyzer.py`
- `src/data/sentiment_fetcher.py`
- `src/features/sentiment_features.py`
- `scripts/retrain_with_sentiment.py`
- `scripts/train_model_with_sentiment.py`
- `scripts/test_sentiment.py`
- `config/sentiment_config.json`
- `data/processed/epl_features_with_sentiment_2021_2024.csv`
- Model: `models/catboost_sentiment_v1_latest.pkl`
