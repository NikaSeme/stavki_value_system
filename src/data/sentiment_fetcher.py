"""
Social media data fetcher for team sentiment analysis.

Supports:
- Twitter API v2 (requires credentials)
- Mock mode for testing (no API needed)
- Caching to respect rate limits
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import sys

# Add project root to sys.path if running as script
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentFetcher:
    """
    Fetch social media data for teams.
    
    Modes:
    - 'mock': Use sample data (no API required)
    - 'twitter': Use Twitter API v2 (requires tweepy + credentials)
    - 'news': Use NewsAPI (requires API key)
    """
    
    def __init__(self, mode='news', config_file='config/sentiment_config.json'):
        """
        Initialize fetcher.
        
        Args:
            mode: 'mock', 'twitter', or 'news'
            config_file: Path to team monitoring config
        """
        self.mode = mode
        self.cache = {}
        self.cache_file = Path('data/cache/sentiment_cache.json')
        
        # Load config
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            logger.warning(f"Config not found, using defaults")
            self.config = {}
        
        # Load cache
        self._load_cache()
        
        # Initialize API client based on mode
        if mode == 'twitter':
            self._init_twitter()
        elif mode == 'news':
            self._init_news()
        else:
            logger.info("Using mock mode (no API required)")
    
    def _init_twitter(self):
        """Initialize Twitter API client."""
        try:
            import tweepy
            
            # Get credentials from env
            from src.config.env import load_env
            env = load_env()
            
            bearer_token = env.get('TWITTER_BEARER_TOKEN')
            if not bearer_token:
                logger.error("TWITTER_BEARER_TOKEN not found in .env")
                logger.info("Falling back to mock mode")
                self.mode = 'mock'
                return
            
            self.twitter_client = tweepy.Client(bearer_token=bearer_token)
            logger.info("✓ Twitter API initialized")
            
        except ImportError:
            logger.error("tweepy not installed: pip install tweepy")
            self.mode = 'mock'
        except Exception as e:
            logger.error(f"Twitter init failed: {e}")
            self.mode = 'mock'
    
    def _init_news(self):
        """Initialize NewsAPI client."""
        try:
            from newsapi import NewsApiClient
            from src.config.env import load_env
            # Try loading env, fallback to production path if needed
            env = load_env()
            
            # Manual fallback read if dotenv fails
            if not env.get('NEWS_API_KEY'):
                try:
                    with open('/etc/stavki/stavki.env') as f:
                        for line in f:
                            if 'NEWS_API_KEY=' in line:
                                env['NEWS_API_KEY'] = line.strip().split('=')[1].replace('"', '').replace("'", "")
                                break
                except Exception:
                    pass

            api_key = env.get('NEWS_API_KEY')
            if not api_key:
                logger.error("NEWS_API_KEY not found in .env or /etc/stavki/stavki.env")
                self.mode = 'mock'
                return
            
            self.news_client = NewsApiClient(api_key=api_key)
            logger.info("✓ NewsAPI initialized")
            
        except Exception as e:
            logger.error(f"NewsAPI init failed (Type: {type(e).__name__}): {e}")
            self.mode = 'mock'
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"NewsAPI init failed: {e}")
            self.mode = 'mock'
    
    def _load_cache(self):
        """Load cached sentiment data."""
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                self.cache = json.load(f)
            logger.info(f"Loaded {len(self.cache)} cached entries")
    
    def _save_cache(self):
        """Save cache to disk."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def fetch_team_posts(
        self, 
        team_name: str, 
        lookback_hours: int = 48,
        max_posts: int = 100
    ) -> List[Dict]:
        """
        Fetch recent posts about a team.
        
        Args:
            team_name: Team name (e.g., "Manchester City")
            lookback_hours: How far back to search
            max_posts: Maximum posts to return
            
        Returns:
            List of posts with text and timestamp
        """
        # Check cache first
        cache_key = f"{team_name}_{lookback_hours}"
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            cache_age = time.time() - cached_data['timestamp']
            
            # Use cache if < 1 hour old
            if cache_age < 3600:
                logger.info(f"Using cached data for {team_name} (age: {cache_age/60:.1f}min)")
                return cached_data['posts']
        
        # Fetch new data
        if self.mode == 'mock':
            posts = self._fetch_mock(team_name, lookback_hours)
        elif self.mode == 'twitter':
            posts = self._fetch_twitter(team_name, lookback_hours, max_posts)
        elif self.mode == 'news':
            posts = self._fetch_news(team_name, lookback_hours, max_posts)
        else:
            posts = []
        
        # Cache results
        self.cache[cache_key] = {
            'timestamp': time.time(),
            'posts': posts
        }
        self._save_cache()
        
        return posts
    
    def _fetch_mock(self, team_name: str, lookback_hours: int) -> List[Dict]:
        """Generate mock social media data for testing."""
        # Sample data based on team
        mock_data = {
            "Manchester City": [
                {"text": "Haaland looking sharp in training ahead of big match", "timestamp": time.time() - 3600},
                {"text": "City's recent form has been exceptional", "timestamp": time.time() - 7200},
                {"text": "De Bruyne returns from injury boost for the team", "timestamp": time.time() - 10800}
            ],
            "Liverpool": [
                {"text": "Salah injury concern ahead of weekend fixture", "timestamp": time.time() - 1800},
                {"text": "Liverpool defense struggling in recent games", "timestamp": time.time() - 5400},
                {"text": "Klopp confident team will bounce back", "timestamp": time.time() - 9000}
            ],
            "Arsenal": [
                {"text": "Arsenal on winning streak looking unstoppable", "timestamp": time.time() - 2400},
                {"text": "Saka fit and ready for crucial match", "timestamp": time.time() - 6000},
                {"text": "Arteta praises team spirit and confidence", "timestamp": time.time() - 12000}
            ]
        }
        
        # Default if team not in mock data
        default = [
            {"text": f"{team_name} preparing for upcoming match", "timestamp": time.time() - 3600},
            {"text": f"Team morale appears good based on recent interviews", "timestamp": time.time() - 7200}
        ]
        
        return mock_data.get(team_name, default)
    
    def _fetch_twitter(self, team_name: str, lookback_hours: int, max_posts: int) -> List[Dict]:
        """Fetch from Twitter API."""
        try:
            # Get team keywords
            team_config = self.config.get('teams', {}).get(team_name, {})
            keywords = team_config.get('keywords', [team_name])
            
            # Build query
            query = ' OR '.join(f'"{kw}"' for kw in keywords[:3])  # Limit to 3 keywords
            query += ' -is:retweet lang:en'  # No retweets, English only
            
            # Time range
            start_time = datetime.utcnow() - timedelta(hours=lookback_hours)
            
            # Search
            tweets = self.twitter_client.search_recent_tweets(
                query=query,
                max_results=min(max_posts, 100),
                start_time=start_time,
                tweet_fields=['created_at']
            )
            
            posts = []
            if tweets.data:
                for tweet in tweets.data:
                    posts.append({
                        'text': tweet.text,
                        'timestamp': tweet.created_at.timestamp()
                    })
            
            logger.info(f"Fetched {len(posts)} tweets for {team_name}")
            return posts
            
        except Exception as e:
            logger.error(f"Twitter fetch failed: {e}")
            return []
    
    def _fetch_news(self, team_name: str, lookback_hours: int, max_posts: int) -> List[Dict]:
        """Fetch from NewsAPI."""
        try:
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(hours=lookback_hours)
            
            # Format dates (YYYY-MM-DDTHH:MM:SS) no microseconds
            from_str = start_date.strftime('%Y-%m-%dT%H:%M:%S')
            to_str = end_date.strftime('%Y-%m-%dT%H:%M:%S')

            # Search
            articles = self.news_client.get_everything(
                q=team_name,
                from_param=from_str,
                to=to_str,
                language='en',
                sort_by='publishedAt',
                page_size=min(max_posts, 100)
            )
            
            posts = []
            for article in articles.get('articles', []):
                posts.append({
                    'text': f"{article['title']} {article['description'] or ''}",
                    'timestamp': datetime.fromisoformat(
                        article['publishedAt'].replace('Z', '+00:00')
                    ).timestamp()
                })
            
            logger.info(f"Fetched {len(posts)} news articles for {team_name}")
            return posts
            
        except Exception as e:
            logger.error(f"NewsAPI fetch failed: {e}")
            return []


def test_fetcher():
    """Test sentiment fetcher."""
    print("=" * 60)
    print("SENTIMENT FETCHER TEST (Default Mode)")
    print("=" * 60)
    
    fetcher = SentimentFetcher()
    
    teams = ["Manchester City", "Liverpool", "Arsenal"]
    
    for team in teams:
        print(f"\n{team}:")
        posts = fetcher.fetch_team_posts(team, lookback_hours=48)
        print(f"  Fetched {len(posts)} posts")
        
        for i, post in enumerate(posts[:2], 1):  # Show first 2
            print(f"  {i}. {post['text'][:60]}...")


if __name__ == '__main__':
    test_fetcher()
