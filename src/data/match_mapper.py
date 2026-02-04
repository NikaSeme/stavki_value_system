#!/usr/bin/env python3
"""
Match ID Mapper.

Maps matches between different API sources:
- The Odds API -> SportMonks fixture ID
- The Odds API -> Betfair market ID
- SportMonks -> Betfair

Uses team name normalization and kickoff time matching.

Usage:
    from src.data.match_mapper import MatchMapper
    
    mapper = MatchMapper()
    ids = mapper.find_sportmonks_fixture("Man City", "Arsenal", "2024-01-15T15:00:00Z")
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..utils.team_normalizer import normalize_team

logger = logging.getLogger(__name__)


@dataclass
class MatchMapping:
    """Mapping between different API sources for a match."""
    odds_api_id: Optional[str] = None
    sportmonks_fixture_id: Optional[int] = None
    betfair_market_id: Optional[str] = None
    home_team: str = ""
    away_team: str = ""
    kickoff: Optional[datetime] = None
    confidence: float = 0.0  # 0-1 confidence in mapping


class MatchMapper:
    """
    Map matches between different data sources.
    
    Uses team name normalization and time window matching
    to link fixtures across APIs.
    """
    
    def __init__(
        self,
        sportmonks_client=None,
        betfair_client=None,
        time_tolerance_minutes: int = 30
    ):
        """
        Initialize mapper with optional API clients.
        
        Args:
            sportmonks_client: SportMonks client for fixture lookup
            betfair_client: Betfair client for market lookup
            time_tolerance_minutes: Max time difference for matching
        """
        self.sportmonks = sportmonks_client
        self.betfair = betfair_client
        self.time_tolerance = timedelta(minutes=time_tolerance_minutes)
        
        # Cache of known mappings
        self._mapping_cache: Dict[str, MatchMapping] = {}
        
        # Pre-loaded fixtures for batch matching
        self._sportmonks_fixtures: List[Dict] = []
        self._betfair_markets: List[Dict] = []
    
    def find_mapping(
        self,
        home_team: str,
        away_team: str,
        kickoff: datetime,
        odds_api_id: Optional[str] = None
    ) -> MatchMapping:
        """
        Find all IDs for a match across different APIs.
        
        Args:
            home_team: Home team name (any format)
            away_team: Away team name (any format)
            kickoff: Match kickoff time
            odds_api_id: Optional The Odds API event ID
            
        Returns:
            MatchMapping with all found IDs
        """
        # Check cache first
        cache_key = self._make_cache_key(home_team, away_team, kickoff)
        if cache_key in self._mapping_cache:
            return self._mapping_cache[cache_key]
        
        # Normalize team names
        home_normalized = normalize_team(home_team)
        away_normalized = normalize_team(away_team)
        
        mapping = MatchMapping(
            odds_api_id=odds_api_id,
            home_team=home_normalized,
            away_team=away_normalized,
            kickoff=kickoff
        )
        
        # Find SportMonks fixture
        if self.sportmonks:
            fixture_id = self._find_sportmonks_fixture(
                home_normalized, away_normalized, kickoff
            )
            if fixture_id:
                mapping.sportmonks_fixture_id = fixture_id
                mapping.confidence = max(mapping.confidence, 0.9)
        
        # Find Betfair market
        if self.betfair:
            market_id = self._find_betfair_market(
                home_normalized, away_normalized, kickoff
            )
            if market_id:
                mapping.betfair_market_id = market_id
                mapping.confidence = max(mapping.confidence, 0.8)
        
        # Cache result
        self._mapping_cache[cache_key] = mapping
        
        return mapping
    
    def _find_sportmonks_fixture(
        self,
        home_team: str,
        away_team: str,
        kickoff: datetime
    ) -> Optional[int]:
        """Find SportMonks fixture ID for a match."""
        # If we have pre-loaded fixtures, search there first
        for fixture in self._sportmonks_fixtures:
            if self._is_match(
                home_team, away_team, kickoff,
                fixture.get("home_team", ""),
                fixture.get("away_team", ""),
                fixture.get("kickoff")
            ):
                return fixture.get("fixture_id")
        
        # Otherwise, query API
        if self.sportmonks is None:
            return None
        
        try:
            date_str = kickoff.strftime("%Y-%m-%d")
            fixtures = self.sportmonks.get_fixtures_by_date(date_str)
            
            for fixture in fixtures:
                fixture_home = normalize_team(fixture.home_team)
                fixture_away = normalize_team(fixture.away_team)
                fixture_time = fixture.kickoff
                
                if self._is_match(
                    home_team, away_team, kickoff,
                    fixture_home, fixture_away, fixture_time
                ):
                    logger.debug(f"Mapped {home_team} vs {away_team} -> fixture {fixture.fixture_id}")
                    return fixture.fixture_id
                    
        except Exception as e:
            logger.warning(f"Failed to find SportMonks fixture: {e}")
        
        return None
    
    def _find_betfair_market(
        self,
        home_team: str,
        away_team: str,
        kickoff: datetime
    ) -> Optional[str]:
        """Find Betfair market ID for a match."""
        # If we have pre-loaded markets, search there first
        for market in self._betfair_markets:
            if self._is_match(
                home_team, away_team, kickoff,
                market.get("home_team", ""),
                market.get("away_team", ""),
                market.get("kickoff")
            ):
                return market.get("market_id")
        
        # Otherwise, query API
        if self.betfair is None:
            return None
        
        try:
            date_str = kickoff.strftime("%Y-%m-%d")
            markets = self.betfair.get_football_markets(date=date_str)
            
            for market in markets:
                # Parse event name (usually "Team A v Team B")
                parts = market.event_name.split(" v ")
                if len(parts) != 2:
                    parts = market.event_name.split(" vs ")
                
                if len(parts) == 2:
                    market_home = normalize_team(parts[0].strip())
                    market_away = normalize_team(parts[1].strip())
                    market_time = market.market_start_time
                    
                    if self._is_match(
                        home_team, away_team, kickoff,
                        market_home, market_away, market_time
                    ):
                        logger.debug(f"Mapped {home_team} vs {away_team} -> market {market.market_id}")
                        return market.market_id
                        
        except Exception as e:
            logger.warning(f"Failed to find Betfair market: {e}")
        
        return None
    
    def _is_match(
        self,
        home1: str, away1: str, time1: datetime,
        home2: str, away2: str, time2: Optional[datetime]
    ) -> bool:
        """Check if two matches are the same."""
        # Normalize both sides
        h1, a1 = normalize_team(home1), normalize_team(away1)
        h2, a2 = normalize_team(home2), normalize_team(away2)
        
        # Check teams match
        if h1 != h2 or a1 != a2:
            return False
        
        # Check time is close enough
        if time2 is None:
            return True
        
        time_diff = abs((time1 - time2).total_seconds())
        return time_diff <= self.time_tolerance.total_seconds()
    
    def _make_cache_key(self, home: str, away: str, kickoff: datetime) -> str:
        """Create unique cache key for a match."""
        h = normalize_team(home)
        a = normalize_team(away)
        t = kickoff.strftime("%Y%m%d%H%M")
        return f"{h}_{a}_{t}"
    
    def preload_fixtures(self, date: str, league_ids: Optional[List[int]] = None):
        """
        Pre-load fixtures for a date to speed up lookups.
        
        Args:
            date: Date string (YYYY-MM-DD)
            league_ids: Optional list of league IDs to filter
        """
        if self.sportmonks is None:
            return
        
        try:
            fixtures = self.sportmonks.get_fixtures_by_date(date, league_ids)
            
            for fixture in fixtures:
                self._sportmonks_fixtures.append({
                    "fixture_id": fixture.fixture_id,
                    "home_team": normalize_team(fixture.home_team),
                    "away_team": normalize_team(fixture.away_team),
                    "kickoff": fixture.kickoff,
                    "league_id": fixture.league_id
                })
            
            logger.info(f"Pre-loaded {len(fixtures)} SportMonks fixtures for {date}")
            
        except Exception as e:
            logger.warning(f"Failed to preload fixtures: {e}")
    
    def preload_markets(self, date: str):
        """
        Pre-load Betfair markets for a date.
        
        Args:
            date: Date string (YYYY-MM-DD)
        """
        if self.betfair is None:
            return
        
        try:
            markets = self.betfair.get_football_markets(date=date)
            
            for market in markets:
                parts = market.event_name.split(" v ")
                if len(parts) != 2:
                    parts = market.event_name.split(" vs ")
                
                if len(parts) == 2:
                    self._betfair_markets.append({
                        "market_id": market.market_id,
                        "home_team": normalize_team(parts[0].strip()),
                        "away_team": normalize_team(parts[1].strip()),
                        "kickoff": market.market_start_time
                    })
            
            logger.info(f"Pre-loaded {len(self._betfair_markets)} Betfair markets for {date}")
            
        except Exception as e:
            logger.warning(f"Failed to preload markets: {e}")
    
    def batch_map(
        self,
        matches: List[Dict],
        preload_date: Optional[str] = None
    ) -> List[MatchMapping]:
        """
        Map multiple matches at once.
        
        Args:
            matches: List of {home_team, away_team, kickoff, [odds_api_id]}
            preload_date: Optional date to preload fixtures/markets
            
        Returns:
            List of MatchMapping objects
        """
        # Preload if date specified
        if preload_date:
            self.preload_fixtures(preload_date)
            self.preload_markets(preload_date)
        
        results = []
        for match in matches:
            mapping = self.find_mapping(
                home_team=match["home_team"],
                away_team=match["away_team"],
                kickoff=match["kickoff"],
                odds_api_id=match.get("odds_api_id")
            )
            results.append(mapping)
        
        # Log stats
        with_sportmonks = sum(1 for m in results if m.sportmonks_fixture_id)
        with_betfair = sum(1 for m in results if m.betfair_market_id)
        
        logger.info(f"Batch mapped {len(results)} matches: "
                   f"{with_sportmonks} with SportMonks, {with_betfair} with Betfair")
        
        return results
    
    def clear_cache(self):
        """Clear all caches."""
        self._mapping_cache.clear()
        self._sportmonks_fixtures.clear()
        self._betfair_markets.clear()


# Convenience function
def map_match(
    home_team: str,
    away_team: str,
    kickoff: datetime,
    sportmonks_client=None,
    betfair_client=None
) -> MatchMapping:
    """
    Map a single match to all available API IDs.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        kickoff: Match kickoff time
        sportmonks_client: Optional SportMonks client
        betfair_client: Optional Betfair client
        
    Returns:
        MatchMapping with found IDs
    """
    mapper = MatchMapper(
        sportmonks_client=sportmonks_client,
        betfair_client=betfair_client
    )
    return mapper.find_mapping(home_team, away_team, kickoff)


# CLI for testing
if __name__ == "__main__":
    from datetime import datetime
    
    print("Match Mapper Test")
    print("=" * 50)
    
    mapper = MatchMapper()
    
    # Test without API clients (should return empty IDs but no errors)
    test_cases = [
        ("Manchester City", "Arsenal", datetime(2024, 3, 31, 16, 30)),
        ("Liverpool", "Man United", datetime(2024, 4, 7, 15, 0)),
        ("Chelsea", "Tottenham", datetime(2024, 4, 15, 20, 0)),
    ]
    
    for home, away, kickoff in test_cases:
        mapping = mapper.find_mapping(home, away, kickoff)
        print(f"\n{home} vs {away} @ {kickoff}")
        print(f"  Home (normalized): {mapping.home_team}")
        print(f"  Away (normalized): {mapping.away_team}")
        print(f"  SportMonks ID: {mapping.sportmonks_fixture_id}")
        print(f"  Betfair ID: {mapping.betfair_market_id}")
        print(f"  Confidence: {mapping.confidence}")
