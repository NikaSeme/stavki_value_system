#!/usr/bin/env python3
"""
Unified Data Collection Script.

Collects data from all configured sources:
- SportMonks: Fixtures, xG, lineups, injuries
- Betfair: Exchange odds, CLV data
- OpenWeatherMap: Weather for venues
- The Odds API: Multi-bookmaker odds

Usage:
    python scripts/collect_all_data.py              # Collect all
    python scripts/collect_all_data.py --fixtures   # Only fixtures
    python scripts/collect_all_data.py --days 3     # Next 3 days
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

from src.data.sportmonks_client import SportMonksClient
from src.data.betfair_client import BetfairClient
from src.data.weather_client import WeatherClient
from src.data.enhanced_features import EnhancedFeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedDataCollector:
    """
    Unified data collector for all STAVKI data sources.
    
    Collects and stores:
    - Match fixtures from SportMonks
    - xG and statistics
    - Lineups and injuries
    - Weather forecasts
    - Exchange odds from Betfair
    - Multi-bookmaker odds
    """
    
    def __init__(
        self,
        data_dir: Path = Path("data/collected"),
        sportmonks_key: Optional[str] = None,
        betfair_key: Optional[str] = None,
        weather_key: Optional[str] = None
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize clients
        self.sportmonks = None
        self.betfair = None
        self.weather = None
        
        sm_key = sportmonks_key or os.getenv("SPORTMONKS_API_KEY")
        if sm_key:
            self.sportmonks = SportMonksClient(api_key=sm_key)
            logger.info("✓ SportMonks client initialized")
        else:
            logger.warning("✗ SportMonks API key not found")
        
        bf_key = betfair_key or os.getenv("BETFAIR_APP_KEY")
        if bf_key:
            self.betfair = BetfairClient(app_key=bf_key)
            logger.info("✓ Betfair client initialized")
        else:
            logger.warning("✗ Betfair API key not found")
        
        wx_key = weather_key or os.getenv("OPENWEATHER_API_KEY")
        if wx_key:
            self.weather = WeatherClient(api_key=wx_key)
            logger.info("✓ Weather client initialized")
        else:
            logger.warning("✗ Weather API key not found")
        
        # Collection metadata
        self.run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.stats = {
            "run_id": self.run_id,
            "started_at": datetime.utcnow().isoformat(),
            "fixtures": 0,
            "xg_stats": 0,
            "lineups": 0,
            "injuries": 0,
            "weather": 0,
            "betfair_markets": 0,
            "errors": []
        }
    
    def collect_fixtures(
        self,
        days: int = 7,
        league_ids: Optional[List[int]] = None
    ) -> List[Dict]:
        """Collect fixtures for the next N days."""
        if not self.sportmonks:
            logger.warning("SportMonks not configured, skipping fixtures")
            return []
        
        logger.info(f"📅 Collecting fixtures for next {days} days...")
        
        all_fixtures = []
        
        for i in range(days):
            date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            try:
                fixtures = self.sportmonks.get_fixtures_by_date(date, league_ids)
                
                for fixture in fixtures:
                    fixture_dict = {
                        "fixture_id": fixture.fixture_id,
                        "league_id": fixture.league_id,
                        "home_team": fixture.home_team,
                        "home_team_id": fixture.home_team_id,
                        "away_team": fixture.away_team,
                        "away_team_id": fixture.away_team_id,
                        "kickoff": fixture.kickoff.isoformat(),
                        "venue": fixture.venue,
                        "round": fixture.round,
                        "status": fixture.status,
                        "collected_at": datetime.utcnow().isoformat()
                    }
                    all_fixtures.append(fixture_dict)
                
                logger.info(f"  {date}: {len(fixtures)} fixtures")
                
            except Exception as e:
                logger.error(f"  {date}: Error - {e}")
                self.stats["errors"].append(f"fixtures_{date}: {e}")
        
        self.stats["fixtures"] = len(all_fixtures)
        
        # Save fixtures
        self._save_json(all_fixtures, f"fixtures_{self.run_id}.json")
        
        return all_fixtures
    
    def collect_xg_stats(
        self,
        fixture_ids: Optional[List[int]] = None,
        team_ids: Optional[List[int]] = None
    ) -> List[Dict]:
        """Collect xG statistics for fixtures or teams."""
        if not self.sportmonks:
            logger.warning("SportMonks not configured, skipping xG stats")
            return []
        
        logger.info("📊 Collecting xG statistics...")
        
        all_stats = []
        
        if fixture_ids:
            for fid in fixture_ids[:20]:  # Limit to avoid rate limits
                try:
                    stats = self.sportmonks.get_fixture_stats(fid)
                    if stats:
                        stats_dict = {
                            "fixture_id": stats.fixture_id,
                            "home_xg": stats.home_xg,
                            "away_xg": stats.away_xg,
                            "home_shots": stats.home_shots,
                            "away_shots": stats.away_shots,
                            "home_shots_on_target": stats.home_shots_on_target,
                            "away_shots_on_target": stats.away_shots_on_target,
                            "home_possession": stats.home_possession,
                            "away_possession": stats.away_possession,
                            "collected_at": datetime.utcnow().isoformat()
                        }
                        all_stats.append(stats_dict)
                except Exception as e:
                    logger.warning(f"  Fixture {fid}: {e}")
        
        self.stats["xg_stats"] = len(all_stats)
        
        if all_stats:
            self._save_json(all_stats, f"xg_stats_{self.run_id}.json")
        
        logger.info(f"  Collected: {len(all_stats)} xG records")
        return all_stats
    
    def collect_lineups(self, fixture_ids: List[int]) -> List[Dict]:
        """Collect lineups for fixtures."""
        if not self.sportmonks:
            logger.warning("SportMonks not configured, skipping lineups")
            return []
        
        logger.info("👥 Collecting lineups...")
        
        all_lineups = []
        
        for fid in fixture_ids[:20]:
            try:
                lineups = self.sportmonks.get_fixture_lineups(fid)
                
                if lineups:
                    lineup_dict = {
                        "fixture_id": fid,
                        "home": {
                            "team_id": lineups.get("home", {}).team_id if "home" in lineups else None,
                            "formation": lineups.get("home", {}).formation if "home" in lineups else None,
                            "starting_xi_count": len(lineups.get("home", {}).starting_xi) if "home" in lineups else 0
                        },
                        "away": {
                            "team_id": lineups.get("away", {}).team_id if "away" in lineups else None,
                            "formation": lineups.get("away", {}).formation if "away" in lineups else None,
                            "starting_xi_count": len(lineups.get("away", {}).starting_xi) if "away" in lineups else 0
                        },
                        "collected_at": datetime.utcnow().isoformat()
                    }
                    all_lineups.append(lineup_dict)
                    
            except Exception as e:
                logger.warning(f"  Fixture {fid}: {e}")
        
        self.stats["lineups"] = len(all_lineups)
        
        if all_lineups:
            self._save_json(all_lineups, f"lineups_{self.run_id}.json")
        
        logger.info(f"  Collected: {len(all_lineups)} lineups")
        return all_lineups
    
    def collect_injuries(self, team_ids: List[int]) -> List[Dict]:
        """Collect injuries for teams."""
        if not self.sportmonks:
            logger.warning("SportMonks not configured, skipping injuries")
            return []
        
        logger.info("🏥 Collecting injuries...")
        
        all_injuries = []
        
        for tid in team_ids[:20]:
            try:
                injuries = self.sportmonks.get_team_injuries(tid)
                
                for injury in injuries:
                    injury_dict = {
                        "player_id": injury.player_id,
                        "player_name": injury.player_name,
                        "team_id": injury.team_id,
                        "team_name": injury.team_name,
                        "type": injury.type,
                        "reason": injury.reason,
                        "expected_return": injury.expected_return.isoformat() if injury.expected_return else None,
                        "collected_at": datetime.utcnow().isoformat()
                    }
                    all_injuries.append(injury_dict)
                    
            except Exception as e:
                logger.warning(f"  Team {tid}: {e}")
        
        self.stats["injuries"] = len(all_injuries)
        
        if all_injuries:
            self._save_json(all_injuries, f"injuries_{self.run_id}.json")
        
        logger.info(f"  Collected: {len(all_injuries)} injuries")
        return all_injuries
    
    def collect_weather(self, fixtures: List[Dict]) -> List[Dict]:
        """Collect weather for fixtures."""
        if not self.weather:
            logger.warning("Weather not configured, skipping")
            return []
        
        logger.info("🌤️ Collecting weather...")
        
        all_weather = []
        
        for fixture in fixtures[:30]:  # Limit API calls
            try:
                home_team = fixture.get("home_team")
                kickoff = datetime.fromisoformat(fixture.get("kickoff"))
                venue = fixture.get("venue")
                
                weather = self.weather.get_match_weather(
                    home_team=home_team,
                    kickoff_time=kickoff,
                    venue=venue
                )
                
                if weather:
                    weather_dict = {
                        "fixture_id": fixture.get("fixture_id"),
                        "home_team": home_team,
                        "temperature": weather.temperature,
                        "feels_like": weather.feels_like,
                        "humidity": weather.humidity,
                        "wind_speed": weather.wind_speed,
                        "precipitation": weather.precipitation,
                        "condition": weather.condition,
                        "weather_score": self.weather.calculate_weather_score(weather),
                        "collected_at": datetime.utcnow().isoformat()
                    }
                    all_weather.append(weather_dict)
                    
            except Exception as e:
                logger.warning(f"  Weather for {fixture.get('home_team')}: {e}")
        
        self.stats["weather"] = len(all_weather)
        
        if all_weather:
            self._save_json(all_weather, f"weather_{self.run_id}.json")
        
        logger.info(f"  Collected: {len(all_weather)} weather forecasts")
        return all_weather
    
    def collect_betfair(self, date: Optional[str] = None) -> List[Dict]:
        """Collect Betfair exchange data."""
        if not self.betfair:
            logger.warning("Betfair not configured, skipping")
            return []
        
        logger.info("📊 Collecting Betfair markets...")
        
        try:
            markets = self.betfair.get_football_markets(date=date)
            
            all_markets = []
            for market in markets[:20]:  # Limit
                try:
                    odds = self.betfair.get_market_odds(market.market_id)
                    liquidity = self.betfair.get_market_liquidity(market.market_id)
                    
                    market_dict = {
                        "market_id": market.market_id,
                        "event_name": market.event_name,
                        "market_start_time": market.market_start_time.isoformat(),
                        "total_matched": liquidity.get("total_matched", 0),
                        "is_liquid": liquidity.get("is_liquid", False),
                        "odds": [
                            {
                                "selection_id": o.selection_id,
                                "back_price": o.back_price,
                                "back_size": o.back_size
                            }
                            for o in odds
                        ],
                        "collected_at": datetime.utcnow().isoformat()
                    }
                    all_markets.append(market_dict)
                except Exception as e:
                    logger.warning(f"  Market {market.market_id}: {e}")
            
            self.stats["betfair_markets"] = len(all_markets)
            
            if all_markets:
                self._save_json(all_markets, f"betfair_{self.run_id}.json")
            
            logger.info(f"  Collected: {len(all_markets)} markets")
            return all_markets
            
        except Exception as e:
            logger.error(f"Betfair collection failed: {e}")
            self.stats["errors"].append(f"betfair: {e}")
            return []
    
    def collect_all(
        self,
        days: int = 7,
        league_ids: Optional[List[int]] = None
    ) -> Dict:
        """Run full collection."""
        logger.info("=" * 60)
        logger.info("🚀 Starting unified data collection")
        logger.info(f"   Run ID: {self.run_id}")
        logger.info(f"   Days: {days}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Fixtures
        fixtures = self.collect_fixtures(days=days, league_ids=league_ids)
        
        # Step 2: Weather for fixtures
        if fixtures:
            self.collect_weather(fixtures)
        
        # Step 3: Betfair markets
        for i in range(min(days, 3)):
            date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            self.collect_betfair(date=date)
        
        # Finalize
        elapsed = time.time() - start_time
        self.stats["finished_at"] = datetime.utcnow().isoformat()
        self.stats["elapsed_seconds"] = round(elapsed, 2)
        
        # Save summary
        self._save_json(self.stats, f"collection_summary_{self.run_id}.json")
        
        # Print summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("📋 COLLECTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Fixtures:       {self.stats['fixtures']}")
        logger.info(f"  xG Stats:       {self.stats['xg_stats']}")
        logger.info(f"  Lineups:        {self.stats['lineups']}")
        logger.info(f"  Injuries:       {self.stats['injuries']}")
        logger.info(f"  Weather:        {self.stats['weather']}")
        logger.info(f"  Betfair:        {self.stats['betfair_markets']}")
        logger.info(f"  Errors:         {len(self.stats['errors'])}")
        logger.info(f"  Elapsed:        {elapsed:.1f}s")
        logger.info("=" * 60)
        
        return self.stats
    
    def _save_json(self, data: any, filename: str):
        """Save data to JSON file."""
        path = self.data_dir / filename
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.debug(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Unified data collection for STAVKI")
    parser.add_argument("--days", type=int, default=7, help="Days to collect (default: 7)")
    parser.add_argument("--fixtures", action="store_true", help="Only collect fixtures")
    parser.add_argument("--weather", action="store_true", help="Only collect weather")
    parser.add_argument("--betfair", action="store_true", help="Only collect Betfair")
    parser.add_argument("--output", type=str, default="data/collected", help="Output directory")
    parser.add_argument("--leagues", type=str, help="Comma-separated league IDs")
    
    args = parser.parse_args()
    
    # Parse league IDs
    league_ids = None
    if args.leagues:
        league_ids = [int(x) for x in args.leagues.split(",")]
    
    # Initialize collector
    collector = UnifiedDataCollector(data_dir=Path(args.output))
    
    # Run collection
    if args.fixtures:
        collector.collect_fixtures(days=args.days, league_ids=league_ids)
    elif args.weather:
        fixtures = collector.collect_fixtures(days=args.days, league_ids=league_ids)
        collector.collect_weather(fixtures)
    elif args.betfair:
        collector.collect_betfair()
    else:
        collector.collect_all(days=args.days, league_ids=league_ids)


if __name__ == "__main__":
    main()
