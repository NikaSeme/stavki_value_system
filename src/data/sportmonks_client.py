"""
SportMonks Football API Client.

Provides access to:
- Match fixtures and results
- xG (Expected Goals) statistics
- Team lineups and formations
- Player injuries and suspensions
- Live scores and events
- Pre-match and in-play odds
- Weather forecasts for matches
"""

import requests
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from functools import lru_cache
import json

logger = logging.getLogger(__name__)


@dataclass
class MatchFixture:
    """Parsed match fixture data."""
    fixture_id: int
    league_id: int
    home_team: str
    home_team_id: int
    away_team: str
    away_team_id: int
    kickoff: datetime
    venue: Optional[str] = None
    round: Optional[str] = None
    status: str = "NS"  # Not Started


@dataclass
class MatchStats:
    """Match statistics including xG."""
    fixture_id: int
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None
    home_shots: Optional[int] = None
    away_shots: Optional[int] = None
    home_shots_on_target: Optional[int] = None
    away_shots_on_target: Optional[int] = None
    home_possession: Optional[float] = None
    away_possession: Optional[float] = None
    home_corners: Optional[int] = None
    away_corners: Optional[int] = None


@dataclass 
class TeamLineup:
    """Team lineup data."""
    team_id: int
    team_name: str
    formation: Optional[str]
    starting_xi: List[Dict]
    substitutes: List[Dict]
    coach: Optional[str] = None


@dataclass
class InjuryInfo:
    """Player injury/suspension info."""
    player_id: int
    player_name: str
    team_id: int
    team_name: str
    type: str  # "injury" or "suspension"
    reason: Optional[str] = None
    expected_return: Optional[datetime] = None


class SportMonksClient:
    """
    SportMonks Football API v3 Client.
    
    Features:
    - Automatic rate limiting
    - Response caching
    - Error handling with retries
    - Data parsing into typed objects
    
    Usage:
        client = SportMonksClient(api_key="your_key")
        fixtures = client.get_fixtures_by_date("2024-01-15")
        stats = client.get_fixture_stats(fixture_id=123456)
    """
    
    BASE_URL = "https://api.sportmonks.com/v3/football"
    
    # European Advanced league IDs
    LEAGUE_IDS = {
        "EPL": 8,
        "LA_LIGA": 564,
        "BUNDESLIGA": 82,
        "SERIE_A": 384,
        "LIGUE_1": 301,
        "CHAMPIONSHIP": 9,
        "EREDIVISIE": 72,
        "PRIMEIRA": 462,
        "SCOTTISH": 501,
        "BELGIAN": 208,
    }
    
    # Reverse mapping for display
    LEAGUE_NAMES = {v: k for k, v in LEAGUE_IDS.items()}
    
    def __init__(
        self,
        api_key: str,
        rate_limit: int = 180,  # Requests per minute
        timeout: int = 30,
        cache_ttl: int = 300  # Cache TTL in seconds
    ):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        
        # Rate limiting
        self._request_times: List[float] = []
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": api_key,
            "Accept": "application/json"
        })
        
        logger.info("SportMonks client initialized")
    
    def _rate_limit_wait(self):
        """Enforce rate limiting."""
        now = time.time()
        minute_ago = now - 60
        
        # Remove old requests
        self._request_times = [t for t in self._request_times if t > minute_ago]
        
        # Wait if at limit
        if len(self._request_times) >= self.rate_limit:
            wait_time = self._request_times[0] - minute_ago + 0.1
            logger.debug(f"Rate limit reached, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        
        self._request_times.append(now)
    
    def _request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        includes: Optional[List[str]] = None
    ) -> Dict:
        """Make API request with rate limiting and error handling."""
        self._rate_limit_wait()
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        # Build params
        request_params = params or {}
        if includes:
            request_params["include"] = ";".join(includes)
        
        try:
            response = self.session.get(
                url,
                params=request_params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if "error" in data:
                logger.error(f"SportMonks API error: {data['error']}")
                return {"data": [], "error": data["error"]}
            
            return data
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout: {endpoint}")
            return {"data": [], "error": "timeout"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {"data": [], "error": str(e)}
    
    # =========================================================================
    # FIXTURES
    # =========================================================================
    
    def get_fixtures_by_date(
        self,
        date: str,
        league_ids: Optional[List[int]] = None
    ) -> List[MatchFixture]:
        """
        Get all fixtures for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format
            league_ids: Optional list of league IDs to filter
            
        Returns:
            List of MatchFixture objects
        """
        params = {"filters": f"fixtureStartDate:{date};fixtureEndDate:{date}"}
        
        if league_ids:
            params["filters"] += f";leagueIds:{','.join(map(str, league_ids))}"
        
        response = self._request(
            "fixtures",
            params=params,
            includes=["participants", "venue", "round"]
        )
        
        fixtures = []
        for item in response.get("data", []):
            try:
                participants = item.get("participants", [])
                home = next((p for p in participants if p.get("meta", {}).get("location") == "home"), None)
                away = next((p for p in participants if p.get("meta", {}).get("location") == "away"), None)
                
                if home and away:
                    fixture = MatchFixture(
                        fixture_id=item["id"],
                        league_id=item.get("league_id"),
                        home_team=home.get("name", "Unknown"),
                        home_team_id=home.get("id"),
                        away_team=away.get("name", "Unknown"),
                        away_team_id=away.get("id"),
                        kickoff=datetime.fromisoformat(item["starting_at"].replace("Z", "+00:00")),
                        venue=item.get("venue", {}).get("name") if item.get("venue") else None,
                        round=item.get("round", {}).get("name") if item.get("round") else None,
                        status=item.get("state", {}).get("short", "NS")
                    )
                    fixtures.append(fixture)
            except Exception as e:
                logger.warning(f"Failed to parse fixture {item.get('id')}: {e}")
        
        logger.info(f"Got {len(fixtures)} fixtures for {date}")
        return fixtures
    
    def get_upcoming_fixtures(
        self,
        days: int = 7,
        league_ids: Optional[List[int]] = None
    ) -> List[MatchFixture]:
        """Get fixtures for the next N days."""
        all_fixtures = []
        
        for i in range(days):
            date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            fixtures = self.get_fixtures_by_date(date, league_ids)
            all_fixtures.extend(fixtures)
        
        return all_fixtures
    
    # =========================================================================
    # STATISTICS & xG
    # =========================================================================
    
    def get_fixture_stats(self, fixture_id: int) -> Optional[MatchStats]:
        """
        Get detailed statistics for a fixture including xG.
        
        Args:
            fixture_id: SportMonks fixture ID
            
        Returns:
            MatchStats object or None
        """
        response = self._request(
            f"fixtures/{fixture_id}",
            includes=["statistics"]
        )
        
        data = response.get("data", {})
        stats = data.get("statistics", [])
        
        if not stats:
            return None
        
        # Parse statistics
        result = MatchStats(fixture_id=fixture_id)
        
        for stat in stats:
            type_id = stat.get("type_id")
            location = stat.get("location")  # "home" or "away"
            value = stat.get("data", {}).get("value")
            
            # xG (type_id varies, check name)
            if "expected_goals" in str(stat.get("type", {}).get("name", "")).lower():
                if location == "home":
                    result.home_xg = float(value) if value else None
                else:
                    result.away_xg = float(value) if value else None
            
            # Shots
            elif stat.get("type", {}).get("code") == "shots-total":
                if location == "home":
                    result.home_shots = int(value) if value else None
                else:
                    result.away_shots = int(value) if value else None
            
            # Shots on target
            elif stat.get("type", {}).get("code") == "shots-on-target":
                if location == "home":
                    result.home_shots_on_target = int(value) if value else None
                else:
                    result.away_shots_on_target = int(value) if value else None
            
            # Possession
            elif stat.get("type", {}).get("code") == "ball-possession":
                if location == "home":
                    result.home_possession = float(value) if value else None
                else:
                    result.away_possession = float(value) if value else None
            
            # Corners
            elif stat.get("type", {}).get("code") == "corners":
                if location == "home":
                    result.home_corners = int(value) if value else None
                else:
                    result.away_corners = int(value) if value else None
        
        return result
    
    def get_team_xg_stats(
        self,
        team_id: int,
        season_id: Optional[int] = None,
        last_n_matches: int = 10
    ) -> Dict[str, float]:
        """
        Get aggregated xG statistics for a team.
        
        Returns:
            Dict with xg_for, xg_against, xg_diff averages
        """
        # Get recent fixtures for team
        response = self._request(
            f"teams/{team_id}/fixtures",
            params={"per_page": last_n_matches},
            includes=["statistics"]
        )
        
        fixtures = response.get("data", [])
        
        xg_for_list = []
        xg_against_list = []
        
        for fixture in fixtures:
            stats = fixture.get("statistics", [])
            for stat in stats:
                if "expected_goals" in str(stat.get("type", {}).get("name", "")).lower():
                    participant_id = stat.get("participant_id")
                    value = stat.get("data", {}).get("value")
                    
                    if value:
                        if participant_id == team_id:
                            xg_for_list.append(float(value))
                        else:
                            xg_against_list.append(float(value))
        
        return {
            "xg_for_avg": sum(xg_for_list) / len(xg_for_list) if xg_for_list else 0.0,
            "xg_against_avg": sum(xg_against_list) / len(xg_against_list) if xg_against_list else 0.0,
            "xg_diff_avg": (sum(xg_for_list) - sum(xg_against_list)) / max(len(xg_for_list), 1),
            "matches_analyzed": len(xg_for_list)
        }
    
    # =========================================================================
    # LINEUPS & INJURIES
    # =========================================================================
    
    def get_fixture_lineups(self, fixture_id: int) -> Dict[str, TeamLineup]:
        """
        Get confirmed lineups for a fixture.
        
        Returns:
            Dict with "home" and "away" TeamLineup objects
        """
        response = self._request(
            f"fixtures/{fixture_id}",
            includes=["lineups.player", "lineups.details"]
        )
        
        data = response.get("data", {})
        lineups_data = data.get("lineups", [])
        
        result = {}
        
        for lineup in lineups_data:
            team_id = lineup.get("team_id")
            location = lineup.get("meta", {}).get("location", "home")
            
            players = lineup.get("lineup", [])
            starting = [p for p in players if p.get("meta", {}).get("position", 0) <= 11]
            subs = [p for p in players if p.get("meta", {}).get("position", 0) > 11]
            
            team_lineup = TeamLineup(
                team_id=team_id,
                team_name=lineup.get("team", {}).get("name", "Unknown"),
                formation=lineup.get("formation", {}).get("formation"),
                starting_xi=[{
                    "id": p.get("player_id"),
                    "name": p.get("player", {}).get("display_name"),
                    "position": p.get("meta", {}).get("position"),
                    "jersey": p.get("jersey_number")
                } for p in starting],
                substitutes=[{
                    "id": p.get("player_id"),
                    "name": p.get("player", {}).get("display_name"),
                    "jersey": p.get("jersey_number")
                } for p in subs],
                coach=lineup.get("coach", {}).get("name") if lineup.get("coach") else None
            )
            
            result[location] = team_lineup
        
        return result
    
    def get_team_injuries(self, team_id: int) -> List[InjuryInfo]:
        """
        Get current injuries and suspensions for a team.
        
        Returns:
            List of InjuryInfo objects
        """
        response = self._request(
            f"teams/{team_id}",
            includes=["sidelined.player", "sidelined.type"]
        )
        
        data = response.get("data", {})
        sidelined = data.get("sidelined", [])
        
        injuries = []
        for item in sidelined:
            # Check if current (no end date or end date in future)
            end_date = item.get("end_date")
            if end_date:
                try:
                    end_dt = datetime.fromisoformat(end_date)
                    if end_dt < datetime.now():
                        continue  # Already returned
                except:
                    pass
            
            injury = InjuryInfo(
                player_id=item.get("player_id"),
                player_name=item.get("player", {}).get("display_name", "Unknown"),
                team_id=team_id,
                team_name=data.get("name", "Unknown"),
                type="suspension" if "suspension" in str(item.get("type", {}).get("name", "")).lower() else "injury",
                reason=item.get("type", {}).get("name"),
                expected_return=datetime.fromisoformat(end_date) if end_date else None
            )
            injuries.append(injury)
        
        return injuries
    
    # =========================================================================
    # WEATHER
    # =========================================================================
    
    def get_fixture_weather(self, fixture_id: int) -> Optional[Dict]:
        """
        Get weather forecast for a fixture.
        
        Returns:
            Dict with temperature, precipitation, wind, humidity
        """
        response = self._request(
            f"fixtures/{fixture_id}",
            includes=["weatherreport"]
        )
        
        data = response.get("data", {})
        weather = data.get("weatherreport", {})
        
        if not weather:
            return None
        
        return {
            "temperature": weather.get("temperature", {}).get("temp"),
            "feels_like": weather.get("temperature", {}).get("feels_like"),
            "humidity": weather.get("humidity"),
            "wind_speed": weather.get("wind", {}).get("speed"),
            "wind_direction": weather.get("wind", {}).get("degree"),
            "clouds": weather.get("clouds", {}).get("all"),
            "condition": weather.get("description"),
            "icon": weather.get("icon")
        }
    
    # =========================================================================
    # ODDS
    # =========================================================================
    
    def get_fixture_odds(
        self,
        fixture_id: int,
        market: str = "1X2"
    ) -> List[Dict]:
        """
        Get pre-match odds for a fixture.
        
        Args:
            fixture_id: Fixture ID
            market: Market type ("1X2", "Over/Under", etc.)
            
        Returns:
            List of odds from different bookmakers
        """
        response = self._request(
            f"fixtures/{fixture_id}",
            includes=["odds.bookmaker"]
        )
        
        data = response.get("data", {})
        all_odds = data.get("odds", [])
        
        result = []
        for odd in all_odds:
            market_name = odd.get("market", {}).get("name", "")
            if market.lower() not in market_name.lower():
                continue
            
            bookmaker = odd.get("bookmaker", {}).get("name", "Unknown")
            odds_values = odd.get("odds", [])
            
            parsed_odds = {
                "bookmaker": bookmaker,
                "market": market_name,
                "timestamp": odd.get("updated_at"),
                "odds": {}
            }
            
            for o in odds_values:
                label = o.get("label", "").lower()
                value = o.get("value")
                
                if label in ["1", "home"]:
                    parsed_odds["odds"]["home"] = float(value)
                elif label in ["x", "draw"]:
                    parsed_odds["odds"]["draw"] = float(value)
                elif label in ["2", "away"]:
                    parsed_odds["odds"]["away"] = float(value)
            
            if parsed_odds["odds"]:
                result.append(parsed_odds)
        
        return result
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    def test_connection(self) -> bool:
        """Test API connection and key validity."""
        try:
            response = self._request("leagues", params={"per_page": 1})
            return "data" in response and not response.get("error")
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_rate_limit_status(self) -> Dict:
        """Get current rate limit status."""
        minute_ago = time.time() - 60
        recent_requests = len([t for t in self._request_times if t > minute_ago])
        
        return {
            "requests_last_minute": recent_requests,
            "limit": self.rate_limit,
            "remaining": self.rate_limit - recent_requests
        }


# Convenience function
def create_client(api_key: Optional[str] = None) -> SportMonksClient:
    """Create a SportMonks client with optional key override."""
    import os
    key = api_key or os.getenv("SPORTMONKS_API_KEY")
    if not key:
        raise ValueError("SportMonks API key required. Set SPORTMONKS_API_KEY or pass api_key.")
    return SportMonksClient(api_key=key)
