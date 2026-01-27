"""
Test sentiment features integration.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.sentiment_features import SentimentFeatureExtractor

def main():
    print("=" * 60)
    print("SENTIMENT FEATURE EXTRACTOR TEST")
    print("=" * 60)
    
    extractor = SentimentFeatureExtractor(mode='news')
    
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
    
    print("\n✅ Sentiment features working!")

if __name__ == '__main__':
    main()
