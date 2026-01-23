"""
Sentiment feature extractor for match prediction.

Adds 6 new features:
- home_sentiment_score, away_sentiment_score
- home_injury_flag, away_injury_flag  
- home_news_volume, away_news_volume
"""

from pathlib import Path
import pandas as pd
from typing import Dict
import logging

from .sentiment_analyzer import SentimentAnalyzer
from ..data.sentiment_fetcher import SentimentFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentFeatureExtractor:
    """Extract sentiment features for matches."""
    
    def __init__(self, mode='mock'):
        """
        Initialize extractor.
        
        Args:
            mode: 'mock', 'twitter', or 'news'
        """
        self.analyzer = SentimentAnalyzer()
        self.fetcher = SentimentFetcher(mode=mode)
        self.mode = mode
    
    def extract_for_team(
        self, 
        team_name: str, 
        lookback_hours: int = 48
    ) -> Dict:
        """
        Extract sentiment features for a team.
        
        Args:
            team_name: Team name
            lookback_hours: How far back to look for posts
            
        Returns:
            Dict with sentiment features
        """
        # Fetch posts
        posts = self.fetcher.fetch_team_posts(team_name, lookback_hours=lookback_hours)
        
        if not posts:
            # No data available - use neutral defaults
            return {
                'sentiment_score': 0.0,
                'injury_flag': 0,
                'news_volume': 0
            }
        
        # Extract texts and timestamps
        texts = [p['text'] for p in posts]
        timestamps = [p['timestamp'] for p in posts]
        
        # Analyze
        agg = self.analyzer.aggregate_team_sentiment(texts, timestamps)
        
        # Normalize news volume (log scale)
        import math
        news_volume = math.log1p(agg['news_count']) / 5.0  # Scale to ~0-1
        
        return {
            'sentiment_score': agg['sentiment_score'],  # -1 to 1
            'injury_flag': int(agg['injury_flag'] or agg['suspension_flag']),  # 0 or 1
            'news_volume': news_volume  # 0 to ~1
        }
    
    def extract_for_match(
        self, 
        home_team: str, 
        away_team: str,
        lookback_hours: int = 48
    ) -> Dict:
        """
        Extract sentiment features for a match.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            lookback_hours: Hours to look back
            
        Returns:
            Dict with 6 features
        """
        # Get features for both teams
        home_features = self.extract_for_team(home_team, lookback_hours)
        away_features = self.extract_for_team(away_team, lookback_hours)
        
        return {
            'home_sentiment_score': home_features['sentiment_score'],
            'away_sentiment_score': away_features['sentiment_score'],
            'home_injury_flag': home_features['injury_flag'],
            'away_injury_flag': away_features['injury_flag'],
            'home_news_volume': home_features['news_volume'],
            'away_news_volume': away_features['news_volume']
        }


def test_extractor():
    """Test sentiment feature extractor."""
    print("=" * 60)
    print("SENTIMENT FEATURE EXTRACTOR TEST")
    print("=" * 60)
    
    extractor = SentimentFeatureExtractor(mode='mock')
    
    # Test match
    home = "Manchester City"
    away = "Liverpool"
    
    print(f"\nMatch: {home} vs {away}")
    
    features = extractor.extract_for_match(home, away, lookback_hours=48)
    
    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
    
    # Interpret
    print("\nInterpretation:")
    if features['home_sentiment_score'] > 0.1:
        print(f"  ✓ {home}: Positive sentiment")
    elif features['home_sentiment_score'] < -0.1:
        print(f"  ⚠️  {home}: Negative sentiment")
    else:
        print(f"  • {home}: Neutral sentiment")
    
    if features['home_injury_flag']:
        print(f"  ⚠️  {home}: Injury/suspension reported")
    
    if features['away_sentiment_score'] > 0.1:
        print(f"  ✓ {away}: Positive sentiment")
    elif features['away_sentiment_score'] < -0.1:
        print(f"  ⚠️  {away}: Negative sentiment")
    else:
        print(f"  • {away}: Neutral sentiment")
    
    if features['away_injury_flag']:
        print(f"  ⚠️  {away}: Injury/suspension reported")


if __name__ == '__main__':
    test_extractor()
