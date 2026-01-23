"""
Sentiment analyzer for social media and news data.

Uses VADER sentiment analysis optimized for social media text.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analyze sentiment from social media posts and news.
    
    Features:
    - VADER sentiment scoring
    - Injury/suspension keyword detection
    - Weighted aggregation by recency
    """
    
    def __init__(self, config_file='config/sentiment_config.json'):
        """Load configuration and initialize VADER."""
        self.vader = SentimentIntensityAnalyzer()
        
        # Load config
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            logger.warning(f"Config not found: {config_file}, using defaults")
            self.config = self._default_config()
        
        self.injury_keywords = self.config['sentiment_keywords']['injury']
        self.suspension_keywords = self.config['sentiment_keywords']['suspension']
        self.positive_keywords = self.config['sentiment_keywords']['positive']
        self.negative_keywords = self.config['sentiment_keywords']['negative']
    
    def _default_config(self):
        """Default configuration if file missing."""
        return {
            'sentiment_keywords': {
                'injury': ['injured', 'injury', 'ruled out', 'out for'],
                'suspension': ['suspended', 'ban', 'red card'],
                'positive': ['confident', 'ready', 'fit', 'strong'],
                'negative': ['struggling', 'poor form', 'crisis', 'worried']
            }
        }
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze single text for sentiment and keywords.
        
        Args:
            text: Social media post or news headline
            
        Returns:
            Dict with sentiment score, injury flag, etc.
        """
        # VADER sentiment
        scores = self.vader.polarity_scores(text)
        
        # Detect keywords
        text_lower = text.lower()
        
        has_injury = any(kw in text_lower for kw in self.injury_keywords)
        has_suspension = any(kw in text_lower for kw in self.suspension_keywords)
        has_positive = any(kw in text_lower for kw in self.positive_keywords)
        has_negative = any(kw in text_lower for kw in self.negative_keywords)
        
        return {
            'sentiment_score': scores['compound'],  # -1 to 1
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'injury_flag': has_injury,
            'suspension_flag': has_suspension,
            'positive_keywords': has_positive,
            'negative_keywords': has_negative
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts."""
        return [self.analyze_text(text) for text in texts]
    
    def aggregate_team_sentiment(
        self, 
        texts: List[str], 
        timestamps: List[float] = None,
        recency_weight: float = 0.3
    ) -> Dict:
        """
        Aggregate sentiment for a team from multiple posts.
        
        Args:
            texts: List of posts/tweets about the team
            timestamps: Unix timestamps (recent posts weighted higher)
            recency_weight: How much to weight recent posts (0-1)
            
        Returns:
            Aggregated sentiment metrics
        """
        if not texts:
            return {
                'sentiment_score': 0.0,
                'injury_flag': False,
                'suspension_flag': False,
                'news_count': 0,
                'negative_ratio': 0.0
            }
        
        # Analyze all texts
        results = self.analyze_batch(texts)
        
        # Simple averaging if no timestamps
        if timestamps is None or recency_weight == 0:
            weights = [1.0] * len(texts)
        else:
            # Weight recent posts more
            max_ts = max(timestamps)
            weights = []
            for ts in timestamps:
                age_factor = (ts - max_ts) / 86400.0  # Days ago (negative)
                weight = 1.0 + recency_weight * age_factor  # Recent = higher
                weights.append(max(weight, 0.1))  # Min weight 0.1
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Weighted aggregation
        avg_sentiment = sum(r['sentiment_score'] * w 
                           for r, w in zip(results, weights))
        
        # Flags (any post mentions)
        injury_flag = any(r['injury_flag'] for r in results)
        suspension_flag = any(r['suspension_flag'] for r in results)
        
        # Negative ratio
        negative_count = sum(1 for r in results if r['sentiment_score'] < -0.1)
        negative_ratio = negative_count / len(results)
        
        return {
            'sentiment_score': float(avg_sentiment),
            'injury_flag': injury_flag,
            'suspension_flag': suspension_flag,
            'news_count': len(texts),
            'negative_ratio': float(negative_ratio)
        }


def test_analyzer():
    """Test sentiment analyzer."""
    analyzer = SentimentAnalyzer()
    
    # Test cases
    test_texts = [
        "Haaland ruled out for 2 weeks with ankle injury",  # Negative + injury
        "Manchester City looking strong ahead of the derby",  # Positive
        "Team morale is low after poor performance",  # Negative
        "Salah fit and ready to play this weekend",  # Positive
        "Player suspended for red card incident"  # Negative + suspension
    ]
    
    print("=" * 60)
    print("SENTIMENT ANALYZER TEST")
    print("=" * 60)
    
    for text in test_texts:
        result = analyzer.analyze_text(text)
        print(f"\nText: {text}")
        print(f"  Sentiment: {result['sentiment_score']:.3f}")
        print(f"  Injury: {result['injury_flag']}")
        print(f"  Suspension: {result['suspension_flag']}")
    
    # Test aggregation
    print("\n" + "=" * 60)
    print("AGGREGATION TEST")
    print("=" * 60)
    
    agg = analyzer.aggregate_team_sentiment(test_texts)
    print(f"Overall sentiment: {agg['sentiment_score']:.3f}")
    print(f"Injury flag: {agg['injury_flag']}")
    print(f"News count: {agg['news_count']}")
    print(f"Negative ratio: {agg['negative_ratio']:.2%}")


if __name__ == '__main__':
    test_analyzer()
