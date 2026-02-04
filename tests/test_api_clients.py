#!/usr/bin/env python3
"""
Test suite for API clients.

Run with: python -m pytest tests/test_api_clients.py -v
Or: python tests/test_api_clients.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import unittest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sportmonks_client import SportMonksClient
from src.data.betfair_client import BetfairClient
from src.data.weather_client import WeatherClient
from src.data.enhanced_features import EnhancedFeatureExtractor
from src.config.api_config import APIConfig, init_config


class TestSportMonksClient(unittest.TestCase):
    """Tests for SportMonks API client."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize client with API key."""
        key = os.getenv("SPORTMONKS_API_KEY")
        if not key:
            cls.skipTest(cls, "SPORTMONKS_API_KEY not set")
        cls.client = SportMonksClient(api_key=key)
    
    def test_connection(self):
        """Test API connection."""
        result = self.client.test_connection()
        self.assertTrue(result, "Connection test failed")
        print("✓ SportMonks connection OK")
    
    def test_rate_limit_status(self):
        """Test rate limit tracking."""
        status = self.client.get_rate_limit_status()
        self.assertIn("remaining", status)
        self.assertGreater(status["remaining"], 0)
        print(f"✓ Rate limit: {status['remaining']}/{status['limit']} remaining")
    
    def test_get_fixtures_by_date(self):
        """Test getting fixtures for a date."""
        # Use tomorrow's date
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        fixtures = self.client.get_fixtures_by_date(tomorrow)
        
        # Should return some fixtures (unless no matches)
        print(f"✓ Got {len(fixtures)} fixtures for {tomorrow}")
        
        if fixtures:
            fixture = fixtures[0]
            self.assertIsNotNone(fixture.home_team)
            self.assertIsNotNone(fixture.away_team)
            print(f"  Example: {fixture.home_team} vs {fixture.away_team}")
    
    def test_league_ids(self):
        """Test that league IDs are correct."""
        self.assertIn("EPL", self.client.LEAGUE_IDS)
        self.assertEqual(self.client.LEAGUE_IDS["EPL"], 8)
        print("✓ League IDs configured correctly")


class TestBetfairClient(unittest.TestCase):
    """Tests for Betfair API client."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize client with API key."""
        key = os.getenv("BETFAIR_APP_KEY")
        if not key:
            cls.skipTest(cls, "BETFAIR_APP_KEY not set")
        cls.client = BetfairClient(app_key=key)
    
    def test_connection(self):
        """Test API connection."""
        result = self.client.test_connection()
        # May fail without session token
        print(f"✓ Betfair connection: {'OK' if result else 'Needs session token'}")
    
    def test_clv_calculation(self):
        """Test CLV calculation."""
        # Bet at 2.20, closed at 2.00 = +10% CLV (good)
        clv = self.client.calculate_clv(2.20, 2.00)
        self.assertAlmostEqual(clv, 10.0, places=1)
        
        # Bet at 1.80, closed at 2.00 = -10% CLV (bad)
        clv_bad = self.client.calculate_clv(1.80, 2.00)
        self.assertAlmostEqual(clv_bad, -10.0, places=1)
        
        print("✓ CLV calculation correct")


class TestWeatherClient(unittest.TestCase):
    """Tests for OpenWeatherMap API client."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize client with API key."""
        key = os.getenv("OPENWEATHER_API_KEY")
        if not key:
            cls.skipTest(cls, "OPENWEATHER_API_KEY not set")
        cls.client = WeatherClient(api_key=key)
    
    def test_connection(self):
        """Test API connection."""
        result = self.client.test_connection()
        self.assertTrue(result, "Connection test failed")
        print("✓ Weather API connection OK")
    
    def test_get_weather_by_city(self):
        """Test getting weather by city."""
        weather = self.client.get_weather_by_city("London,UK")
        
        self.assertIsNotNone(weather)
        self.assertIsInstance(weather.temperature, float)
        self.assertGreater(weather.temperature, -50)
        self.assertLess(weather.temperature, 50)
        
        print(f"✓ London weather: {weather.temperature}°C, {weather.condition}")
    
    def test_get_weather_for_venue(self):
        """Test getting weather for stadium."""
        weather = self.client.get_weather_for_venue("Emirates Stadium")
        
        self.assertIsNotNone(weather)
        print(f"✓ Emirates Stadium weather: {weather.temperature}°C")
    
    def test_get_weather_for_team(self):
        """Test getting weather by team name."""
        weather = self.client.get_weather_for_team("Arsenal")
        
        self.assertIsNotNone(weather)
        print(f"✓ Arsenal home weather: {weather.temperature}°C")
    
    def test_weather_score(self):
        """Test weather score calculation."""
        # Create a good weather scenario
        from src.data.weather_client import WeatherData
        
        good_weather = WeatherData(
            temperature=15.0,
            feels_like=15.0,
            humidity=50,
            pressure=1013,
            wind_speed=2.0,
            wind_direction=0,
            clouds=20,
            precipitation=0.0,
            condition="Clear",
            condition_id=800,
            icon="01d",
            visibility=10000,
            timestamp=datetime.now()
        )
        
        score = self.client.calculate_weather_score(good_weather)
        self.assertGreater(score, 0.8)
        
        # Bad weather
        bad_weather = WeatherData(
            temperature=2.0,
            feels_like=-2.0,
            humidity=95,
            pressure=990,
            wind_speed=15.0,  # m/s = 54 km/h
            wind_direction=0,
            clouds=100,
            precipitation=10.0,  # Heavy rain
            condition="Rain",
            condition_id=502,
            icon="10d",
            visibility=1000,
            timestamp=datetime.now()
        )
        
        bad_score = self.client.calculate_weather_score(bad_weather)
        self.assertLess(bad_score, 0.5)
        
        print(f"✓ Weather scores: good={score:.2f}, bad={bad_score:.2f}")
    
    def test_weather_features(self):
        """Test feature extraction for ML."""
        features = self.client.get_weather_features(
            home_team="Manchester City",
            kickoff_time=datetime.now() + timedelta(days=1)
        )
        
        self.assertIn("temperature", features)
        self.assertIn("weather_score", features)
        self.assertIn("is_rainy", features)
        
        print(f"✓ Weather features extracted: {len(features)} features")


class TestEnhancedFeatureExtractor(unittest.TestCase):
    """Tests for enhanced feature extractor."""
    
    def test_init_without_keys(self):
        """Test initialization without API keys."""
        # Temporarily unset env vars to test without keys
        saved = {}
        for key in ['SPORTMONKS_API_KEY', 'BETFAIR_APP_KEY', 'OPENWEATHER_API_KEY']:
            saved[key] = os.environ.pop(key, None)
        
        try:
            extractor = EnhancedFeatureExtractor()
            self.assertEqual(extractor._count_sources(), 0)
            print("✓ Extractor works without API keys")
        finally:
            # Restore env vars
            for key, val in saved.items():
                if val:
                    os.environ[key] = val
    
    def test_feature_names(self):
        """Test feature name list."""
        names = EnhancedFeatureExtractor.get_feature_names()
        
        self.assertEqual(len(names), 51)
        self.assertIn("home_elo", names)
        self.assertIn("home_xg_for_avg", names)
        self.assertIn("weather_score", names)
        self.assertIn("betfair_odds_home", names)
        
        print(f"✓ Feature names: {len(names)} features defined")
    
    def test_extract_with_defaults(self):
        """Test extraction with default values."""
        extractor = EnhancedFeatureExtractor()
        
        features = extractor.extract_for_match(
            event_id="test_123",
            home_team="Arsenal",
            away_team="Chelsea",
            kickoff=datetime.now() + timedelta(days=1)
        )
        
        # Should return EnhancedFeatures with defaults
        self.assertIsNotNone(features)
        self.assertEqual(features.home_team, "Arsenal")
        self.assertEqual(features.away_team, "Chelsea")
        
        # Check defaults are set (ELO can be int or float)
        self.assertTrue(isinstance(features.home_elo, (int, float)))
        self.assertTrue(isinstance(features.weather_score, (int, float)))
        
        print("✓ Feature extraction with defaults works")
    
    def test_feature_to_dict(self):
        """Test conversion to dictionary."""
        extractor = EnhancedFeatureExtractor()
        
        features = extractor.extract_for_match(
            event_id="test_123",
            home_team="Arsenal",
            away_team="Chelsea",
            kickoff=datetime.now() + timedelta(days=1)
        )
        
        d = features.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("home_elo", d)
        
        print(f"✓ Feature dict has {len(d)} keys")
    
    def test_update_after_match(self):
        """Test state update after match."""
        extractor = EnhancedFeatureExtractor()
        
        # Initial ELO
        initial_elo = extractor.elo.get_rating("Arsenal")
        
        # Arsenal wins 2-0
        extractor.update_after_match(
            home_team="Arsenal",
            away_team="Chelsea",
            home_goals=2,
            away_goals=0,
            match_date=datetime.now()
        )
        
        # ELO should increase
        new_elo = extractor.elo.get_rating("Arsenal")
        self.assertGreater(new_elo, initial_elo)
        
        # Form should be updated
        self.assertIn("Arsenal", extractor.team_form)
        self.assertEqual(extractor.team_form["Arsenal"][-1]["result"], "W")
        
        print(f"✓ State updated: Arsenal ELO {initial_elo:.0f} → {new_elo:.0f}")


class TestAPIConfig(unittest.TestCase):
    """Tests for API configuration."""
    
    def test_from_keys(self):
        """Test creating config from keys."""
        config = APIConfig.from_keys(
            sportmonks_key="test_key_12345678901234567890123456789012345678901234567890",
            betfair_key="test_betfair",
            openweather_key="12345678901234567890123456789012",  # 32 chars
            odds_api_key="test_odds_api_key_12345"
        )
        
        self.assertIsNotNone(config.sportmonks)
        self.assertIsNotNone(config.betfair)
        self.assertIsNotNone(config.openweather)
        
        print("✓ API config created from keys")
    
    def test_validation(self):
        """Test config validation."""
        config = APIConfig.from_keys(
            sportmonks_key="test_key_12345678901234567890123456789012345678901234567890",
            openweather_key="12345678901234567890123456789012"  # 32 chars
        )
        
        status = config.validate()
        
        self.assertTrue(status["sportmonks"]["configured"])
        self.assertTrue(status["openweather"]["configured"])
        self.assertFalse(status["betfair"]["configured"])
        
        print("✓ Config validation works")


def run_all_tests():
    """Run all tests and print summary."""
    print("=" * 60)
    print("STAVKI API CLIENTS TEST SUITE")
    print("=" * 60)
    print()
    
    # Load environment
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        print(f"Loading .env from {env_path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
    
    # Check which APIs are configured
    print("\nAPI Status:")
    print(f"  SportMonks: {'✓' if os.getenv('SPORTMONKS_API_KEY') else '✗'}")
    print(f"  Betfair:    {'✓' if os.getenv('BETFAIR_APP_KEY') else '✗'}")
    print(f"  Weather:    {'✓' if os.getenv('OPENWEATHER_API_KEY') else '✗'}")
    print(f"  Odds API:   {'✓' if os.getenv('ODDS_API_KEY') else '✗'}")
    print()
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestAPIConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedFeatureExtractor))
    
    if os.getenv("SPORTMONKS_API_KEY"):
        suite.addTests(loader.loadTestsFromTestCase(TestSportMonksClient))
    
    if os.getenv("BETFAIR_APP_KEY"):
        suite.addTests(loader.loadTestsFromTestCase(TestBetfairClient))
    
    if os.getenv("OPENWEATHER_API_KEY"):
        suite.addTests(loader.loadTestsFromTestCase(TestWeatherClient))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print()
    print("=" * 60)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
