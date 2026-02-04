"""
OpenWeatherMap API Client.

Provides weather data for match venues:
- Temperature
- Precipitation (rain/snow)
- Wind speed and direction
- Humidity
- Cloud cover

Weather can affect match outcomes:
- Heavy rain: fewer goals, more slips
- Strong wind: less accurate passes/shots
- Extreme temperatures: player fatigue
"""

import requests
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class WeatherData:
    """Weather data for a location."""
    temperature: float          # Celsius
    feels_like: float           # Celsius  
    humidity: int               # Percentage
    pressure: int               # hPa
    wind_speed: float           # m/s
    wind_direction: int         # Degrees
    clouds: int                 # Percentage
    precipitation: float        # mm in last hour
    condition: str              # Weather condition
    condition_id: int           # Weather condition code
    icon: str                   # Weather icon code
    visibility: int             # Meters
    timestamp: datetime


class WeatherClient:
    """
    OpenWeatherMap API Client.
    
    Provides weather data for football match venues.
    
    Usage:
        client = WeatherClient(api_key="your_key")
        weather = client.get_weather_by_coords(51.55, -0.11)  # Emirates
        forecast = client.get_forecast_by_city("Manchester")
    """
    
    BASE_URL = "https://api.openweathermap.org/data/2.5"
    
    # Stadium coordinates for major venues
    STADIUM_COORDS = {
        # Premier League
        "Emirates Stadium": (51.5549, -0.1084),
        "Anfield": (53.4308, -2.9608),
        "Old Trafford": (53.4631, -2.2913),
        "Etihad Stadium": (53.4831, -2.2004),
        "Stamford Bridge": (51.4817, -0.1910),
        "Tottenham Hotspur Stadium": (51.6042, -0.0662),
        "London Stadium": (51.5387, -0.0166),
        "St James' Park": (54.9756, -1.6216),
        "Villa Park": (52.5092, -1.8847),
        "Goodison Park": (53.4388, -2.9664),
        
        # La Liga
        "Santiago Bernabéu": (40.4531, -3.6883),
        "Camp Nou": (41.3809, 2.1228),
        "Wanda Metropolitano": (40.4362, -3.5995),
        "Mestalla": (39.4745, -0.3583),
        
        # Bundesliga
        "Allianz Arena": (48.2188, 11.6247),
        "Signal Iduna Park": (51.4926, 7.4518),
        "Olympiastadion Berlin": (52.5147, 13.2395),
        
        # Serie A
        "San Siro": (45.4781, 9.1240),
        "Stadio Olimpico": (41.9340, 12.4547),
        "Allianz Stadium": (45.1096, 7.6412),
        
        # Ligue 1
        "Parc des Princes": (48.8414, 2.2530),
        "Groupama Stadium": (45.7654, 4.9822),
        "Stade Vélodrome": (43.2697, 5.3958),
    }
    
    # Team to stadium mapping
    TEAM_STADIUMS = {
        "Arsenal": "Emirates Stadium",
        "Liverpool": "Anfield",
        "Manchester United": "Old Trafford",
        "Manchester City": "Etihad Stadium",
        "Chelsea": "Stamford Bridge",
        "Tottenham": "Tottenham Hotspur Stadium",
        "West Ham": "London Stadium",
        "Newcastle": "St James' Park",
        "Aston Villa": "Villa Park",
        "Everton": "Goodison Park",
        "Real Madrid": "Santiago Bernabéu",
        "Barcelona": "Camp Nou",
        "Atletico Madrid": "Wanda Metropolitano",
        "Bayern Munich": "Allianz Arena",
        "Borussia Dortmund": "Signal Iduna Park",
        "AC Milan": "San Siro",
        "Inter Milan": "San Siro",
        "Juventus": "Allianz Stadium",
        "Roma": "Stadio Olimpico",
        "Lazio": "Stadio Olimpico",
        "PSG": "Parc des Princes",
        "Lyon": "Groupama Stadium",
        "Marseille": "Stade Vélodrome",
    }
    
    def __init__(
        self,
        api_key: str,
        timeout: int = 15,
        units: str = "metric"
    ):
        """
        Initialize weather client.
        
        Args:
            api_key: OpenWeatherMap API key
            timeout: Request timeout in seconds
            units: Temperature units (metric, imperial, kelvin)
        """
        self.api_key = api_key
        self.timeout = timeout
        self.units = units
        
        self.session = requests.Session()
        
        logger.info("Weather client initialized")
    
    def _request(self, endpoint: str, params: Dict) -> Dict:
        """Make API request."""
        url = f"{self.BASE_URL}/{endpoint}"
        params["appid"] = self.api_key
        params["units"] = self.units
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout: {endpoint}")
            return {"error": "timeout"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {"error": str(e)}
    
    def _parse_weather(self, data: Dict) -> Optional[WeatherData]:
        """Parse weather response into WeatherData object."""
        try:
            main = data.get("main", {})
            wind = data.get("wind", {})
            weather = data.get("weather", [{}])[0]
            rain = data.get("rain", {})
            
            return WeatherData(
                temperature=main.get("temp", 15.0),
                feels_like=main.get("feels_like", 15.0),
                humidity=main.get("humidity", 50),
                pressure=main.get("pressure", 1013),
                wind_speed=wind.get("speed", 0.0),
                wind_direction=wind.get("deg", 0),
                clouds=data.get("clouds", {}).get("all", 0),
                precipitation=rain.get("1h", 0.0),
                condition=weather.get("main", "Clear"),
                condition_id=weather.get("id", 800),
                icon=weather.get("icon", "01d"),
                visibility=data.get("visibility", 10000),
                timestamp=datetime.fromtimestamp(data.get("dt", time.time()))
            )
        except Exception as e:
            logger.error(f"Failed to parse weather data: {e}")
            return None
    
    # =========================================================================
    # CURRENT WEATHER
    # =========================================================================
    
    def get_weather_by_coords(
        self,
        lat: float,
        lon: float
    ) -> Optional[WeatherData]:
        """
        Get current weather by coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            WeatherData object or None
        """
        data = self._request("weather", {"lat": lat, "lon": lon})
        
        if "error" in data:
            return None
        
        return self._parse_weather(data)
    
    def get_weather_by_city(self, city: str) -> Optional[WeatherData]:
        """
        Get current weather by city name.
        
        Args:
            city: City name (e.g., "London,UK")
            
        Returns:
            WeatherData object or None
        """
        data = self._request("weather", {"q": city})
        
        if "error" in data:
            return None
        
        return self._parse_weather(data)
    
    def get_weather_for_venue(self, venue_name: str) -> Optional[WeatherData]:
        """
        Get weather for a known stadium.
        
        Args:
            venue_name: Stadium name (must be in STADIUM_COORDS)
            
        Returns:
            WeatherData object or None
        """
        coords = self.STADIUM_COORDS.get(venue_name)
        if not coords:
            logger.warning(f"Unknown venue: {venue_name}")
            return None
        
        return self.get_weather_by_coords(coords[0], coords[1])
    
    def get_weather_for_team(self, team_name: str) -> Optional[WeatherData]:
        """
        Get weather for a team's home stadium.
        
        Args:
            team_name: Team name
            
        Returns:
            WeatherData object or None
        """
        # Try exact match
        stadium = self.TEAM_STADIUMS.get(team_name)
        
        # Try partial match
        if not stadium:
            for team, stadia in self.TEAM_STADIUMS.items():
                if team.lower() in team_name.lower() or team_name.lower() in team.lower():
                    stadium = stadia
                    break
        
        if not stadium:
            logger.warning(f"Unknown team: {team_name}")
            return None
        
        return self.get_weather_for_venue(stadium)
    
    # =========================================================================
    # FORECAST
    # =========================================================================
    
    def get_forecast_by_coords(
        self,
        lat: float,
        lon: float,
        target_time: Optional[datetime] = None
    ) -> Optional[WeatherData]:
        """
        Get weather forecast for a specific time.
        
        Args:
            lat: Latitude
            lon: Longitude
            target_time: Target time (default: now)
            
        Returns:
            WeatherData for closest forecast time
        """
        data = self._request("forecast", {"lat": lat, "lon": lon})
        
        if "error" in data or "list" not in data:
            return None
        
        forecasts = data["list"]
        
        if target_time is None:
            target_time = datetime.now()
        
        # Find closest forecast
        closest = None
        min_diff = float("inf")
        
        for item in forecasts:
            forecast_time = datetime.fromtimestamp(item["dt"])
            diff = abs((forecast_time - target_time).total_seconds())
            
            if diff < min_diff:
                min_diff = diff
                closest = item
        
        if closest:
            return self._parse_weather(closest)
        
        return None
    
    def get_match_weather(
        self,
        home_team: str,
        kickoff_time: datetime,
        venue: Optional[str] = None
    ) -> Optional[WeatherData]:
        """
        Get weather forecast for a match.
        
        Args:
            home_team: Home team name
            kickoff_time: Match kickoff time
            venue: Optional venue name
            
        Returns:
            WeatherData for match
        """
        # Try venue first
        if venue:
            coords = self.STADIUM_COORDS.get(venue)
            if coords:
                return self.get_forecast_by_coords(
                    coords[0], coords[1], kickoff_time
                )
        
        # Fall back to team stadium
        stadium = self.TEAM_STADIUMS.get(home_team)
        if stadium:
            coords = self.STADIUM_COORDS.get(stadium)
            if coords:
                return self.get_forecast_by_coords(
                    coords[0], coords[1], kickoff_time
                )
        
        logger.warning(f"Could not find venue for {home_team}")
        return None
    
    # =========================================================================
    # FEATURE EXTRACTION
    # =========================================================================
    
    def calculate_weather_score(self, weather: WeatherData) -> float:
        """
        Calculate composite weather score (0-1).
        
        0 = Extreme conditions (heavy rain, strong wind)
        1 = Perfect conditions (dry, calm, moderate temp)
        
        Args:
            weather: WeatherData object
            
        Returns:
            Score between 0 and 1
        """
        score = 1.0
        
        # Temperature penalty (optimal: 10-20°C)
        temp = weather.temperature
        if temp < 0:
            score -= 0.3
        elif temp < 5:
            score -= 0.1
        elif temp > 30:
            score -= 0.2
        elif temp > 25:
            score -= 0.1
        
        # Wind penalty (km/h equivalent)
        wind_kmh = weather.wind_speed * 3.6
        if wind_kmh > 40:
            score -= 0.3
        elif wind_kmh > 25:
            score -= 0.2
        elif wind_kmh > 15:
            score -= 0.1
        
        # Precipitation penalty
        if weather.precipitation > 5:
            score -= 0.3
        elif weather.precipitation > 2:
            score -= 0.2
        elif weather.precipitation > 0:
            score -= 0.1
        
        # Visibility penalty
        if weather.visibility < 1000:
            score -= 0.2
        elif weather.visibility < 5000:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def get_weather_features(
        self,
        home_team: str,
        kickoff_time: datetime,
        venue: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get weather features for ML model.
        
        Returns:
            Dict with weather features
        """
        weather = self.get_match_weather(home_team, kickoff_time, venue)
        
        if weather is None:
            # Return default values
            return {
                "temperature": 15.0,
                "precipitation": 0.0,
                "wind_speed": 0.0,
                "humidity": 50,
                "weather_score": 0.8,
                "is_rainy": 0,
                "is_windy": 0,
                "is_extreme_temp": 0
            }
        
        return {
            "temperature": weather.temperature,
            "precipitation": weather.precipitation,
            "wind_speed": weather.wind_speed * 3.6,  # Convert to km/h
            "humidity": weather.humidity,
            "weather_score": self.calculate_weather_score(weather),
            "is_rainy": 1 if weather.precipitation > 0.5 else 0,
            "is_windy": 1 if weather.wind_speed * 3.6 > 20 else 0,
            "is_extreme_temp": 1 if weather.temperature < 5 or weather.temperature > 28 else 0
        }
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            data = self._request("weather", {"q": "London,UK"})
            return "error" not in data
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def add_stadium(self, name: str, lat: float, lon: float):
        """Add a custom stadium to the lookup."""
        self.STADIUM_COORDS[name] = (lat, lon)
    
    def add_team_stadium(self, team: str, stadium: str):
        """Map a team to a stadium."""
        self.TEAM_STADIUMS[team] = stadium


# Convenience function
def create_client(api_key: Optional[str] = None) -> WeatherClient:
    """Create a Weather client with optional key override."""
    import os
    key = api_key or os.getenv("OPENWEATHER_API_KEY")
    if not key:
        raise ValueError("OpenWeather API key required. Set OPENWEATHER_API_KEY or pass api_key.")
    return WeatherClient(api_key=key)
