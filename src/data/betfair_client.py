"""
Betfair Exchange API Client.

Provides access to:
- Market odds (true market prices without bookmaker margin)
- Market volumes and liquidity
- Historical odds for CLV tracking
- Market depth

Note: This uses the free tier which has some limitations.
For full features, Betfair Exchange API subscription is needed.
"""

import requests
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class BetfairMarket:
    """Betfair market data."""
    market_id: str
    event_id: str
    event_name: str
    market_name: str
    market_start_time: datetime
    total_matched: float  # Total money matched
    status: str


@dataclass
class BetfairOdds:
    """Betfair odds for a selection."""
    selection_id: int
    selection_name: str
    back_price: Optional[float]  # Best back price
    back_size: Optional[float]   # Available at back price
    lay_price: Optional[float]   # Best lay price
    lay_size: Optional[float]    # Available at lay price
    last_traded: Optional[float] # Last traded price
    total_matched: float         # Total matched on this selection


class BetfairClient:
    """
    Betfair Exchange API Client.
    
    Provides true market odds without bookmaker margin.
    Used for:
    - CLV (Closing Line Value) benchmarking
    - Liquidity checking
    - Sharp money detection
    
    Usage:
        client = BetfairClient(app_key="your_key")
        markets = client.get_football_markets(date="2024-01-15")
        odds = client.get_market_odds(market_id="1.23456789")
    """
    
    # API endpoints
    BETTING_URL = "https://api.betfair.com/exchange/betting/rest/v1.0"
    ACCOUNTS_URL = "https://api.betfair.com/exchange/account/rest/v1.0"
    
    # Football event type
    FOOTBALL_EVENT_TYPE = "1"
    
    # Competition IDs for major leagues
    COMPETITION_IDS = {
        "EPL": 10932509,
        "LA_LIGA": 117,
        "BUNDESLIGA": 59,
        "SERIE_A": 81,
        "LIGUE_1": 55,
        "CHAMPIONSHIP": 7129730,
    }
    
    def __init__(
        self,
        app_key: str,
        session_token: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize Betfair client.
        
        Args:
            app_key: Betfair application key
            session_token: Optional session token for authenticated requests
            timeout: Request timeout in seconds
        """
        self.app_key = app_key
        self.session_token = session_token
        self.timeout = timeout
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "X-Application": app_key,
            "Accept": "application/json",
            "Content-Type": "application/json"
        })
        
        if session_token:
            self.session.headers["X-Authentication"] = session_token
        
        logger.info("Betfair client initialized")
    
    def _request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        method: str = "POST"
    ) -> Dict:
        """Make API request."""
        url = f"{self.BETTING_URL}/{endpoint}/"
        
        try:
            if method == "POST":
                response = self.session.post(
                    url,
                    json=params or {},
                    timeout=self.timeout
                )
            else:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout
                )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout: {endpoint}")
            return {"error": "timeout"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {"error": str(e)}
    
    # =========================================================================
    # MARKETS
    # =========================================================================
    
    def get_football_markets(
        self,
        date: Optional[str] = None,
        competition_ids: Optional[List[int]] = None,
        market_type: str = "MATCH_ODDS"
    ) -> List[BetfairMarket]:
        """
        Get football markets for a date.
        
        Args:
            date: Date in YYYY-MM-DD format (default: today)
            competition_ids: Filter by competition
            market_type: Market type (MATCH_ODDS, OVER_UNDER_25, etc.)
            
        Returns:
            List of BetfairMarket objects
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Build filter
        market_filter = {
            "eventTypeIds": [self.FOOTBALL_EVENT_TYPE],
            "marketTypeCodes": [market_type],
            "marketStartTime": {
                "from": f"{date}T00:00:00Z",
                "to": f"{date}T23:59:59Z"
            }
        }
        
        if competition_ids:
            market_filter["competitionIds"] = [str(c) for c in competition_ids]
        
        params = {
            "filter": market_filter,
            "maxResults": 200,
            "marketProjection": ["EVENT", "MARKET_START_TIME", "RUNNER_DESCRIPTION"]
        }
        
        response = self._request("listMarketCatalogue", params)
        
        if "error" in response:
            return []
        
        markets = []
        for item in response:
            try:
                market = BetfairMarket(
                    market_id=item["marketId"],
                    event_id=str(item.get("event", {}).get("id", "")),
                    event_name=item.get("event", {}).get("name", "Unknown"),
                    market_name=item.get("marketName", "Match Odds"),
                    market_start_time=datetime.fromisoformat(
                        item["marketStartTime"].replace("Z", "+00:00")
                    ),
                    total_matched=0.0,  # Updated when getting book
                    status="OPEN"
                )
                markets.append(market)
            except Exception as e:
                logger.warning(f"Failed to parse market: {e}")
        
        logger.info(f"Got {len(markets)} markets for {date}")
        return markets
    
    def get_market_odds(self, market_id: str) -> List[BetfairOdds]:
        """
        Get current odds for a market.
        
        Args:
            market_id: Betfair market ID
            
        Returns:
            List of BetfairOdds for each selection
        """
        params = {
            "marketIds": [market_id],
            "priceProjection": {
                "priceData": ["EX_BEST_OFFERS", "EX_TRADED"],
                "virtualise": True
            }
        }
        
        response = self._request("listMarketBook", params)
        
        if "error" in response or not response:
            return []
        
        market_book = response[0] if response else {}
        runners = market_book.get("runners", [])
        
        odds_list = []
        for runner in runners:
            back_prices = runner.get("ex", {}).get("availableToBack", [])
            lay_prices = runner.get("ex", {}).get("availableToLay", [])
            
            best_back = back_prices[0] if back_prices else {}
            best_lay = lay_prices[0] if lay_prices else {}
            
            odds = BetfairOdds(
                selection_id=runner["selectionId"],
                selection_name=self._get_selection_name(market_id, runner["selectionId"]),
                back_price=best_back.get("price"),
                back_size=best_back.get("size"),
                lay_price=best_lay.get("price"),
                lay_size=best_lay.get("size"),
                last_traded=runner.get("lastPriceTraded"),
                total_matched=runner.get("totalMatched", 0.0)
            )
            odds_list.append(odds)
        
        return odds_list
    
    def _get_selection_name(self, market_id: str, selection_id: int) -> str:
        """Get human-readable selection name."""
        # Cache this in production
        # For now, return selection ID as string
        # Full implementation would call listMarketCatalogue with RUNNER_DESCRIPTION
        return str(selection_id)
    
    # =========================================================================
    # CLV TRACKING
    # =========================================================================
    
    def get_closing_odds(self, market_id: str) -> Optional[Dict]:
        """
        Get closing odds for a market.
        
        Note: This should be called just before market closes (kick-off).
        
        Returns:
            Dict with selection_id -> closing_price
        """
        odds = self.get_market_odds(market_id)
        
        return {
            o.selection_id: {
                "back": o.back_price,
                "lay": o.lay_price,
                "last_traded": o.last_traded,
                "matched": o.total_matched
            }
            for o in odds
        }
    
    def calculate_clv(
        self,
        bet_odds: float,
        closing_odds: float
    ) -> float:
        """
        Calculate Closing Line Value.
        
        CLV% = (bet_odds - closing_odds) / closing_odds * 100
        
        Positive = beat the closing line (good)
        Negative = worse than closing (bad)
        
        Args:
            bet_odds: Odds at which bet was placed
            closing_odds: Closing odds from Betfair
            
        Returns:
            CLV percentage
        """
        if closing_odds <= 1.0:
            return 0.0
        
        return ((bet_odds - closing_odds) / closing_odds) * 100
    
    # =========================================================================
    # LIQUIDITY
    # =========================================================================
    
    def get_market_liquidity(self, market_id: str) -> Dict:
        """
        Get liquidity information for a market.
        
        Returns:
            Dict with total_matched, back_liquidity, lay_liquidity
        """
        odds = self.get_market_odds(market_id)
        
        total_matched = sum(o.total_matched for o in odds)
        back_liquidity = sum(o.back_size or 0 for o in odds)
        lay_liquidity = sum(o.lay_size or 0 for o in odds)
        
        return {
            "total_matched": total_matched,
            "back_liquidity": back_liquidity,
            "lay_liquidity": lay_liquidity,
            "is_liquid": total_matched > 10000,  # £10k threshold
            "selections": len(odds)
        }
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    def extract_match_odds(self, market_id: str) -> Dict[str, float]:
        """
        Extract 1X2 odds in simple format.
        
        Returns:
            Dict with home, draw, away odds
        """
        odds = self.get_market_odds(market_id)
        
        # Betfair selection IDs:
        # Usually: lowest = home, highest = draw, middle = away
        # But this varies - need runner description for accuracy
        
        sorted_odds = sorted(odds, key=lambda x: x.selection_id)
        
        result = {"home": None, "draw": None, "away": None}
        
        if len(sorted_odds) >= 3:
            # Common pattern
            result["home"] = sorted_odds[0].back_price
            result["draw"] = sorted_odds[2].back_price  # Draw usually highest ID
            result["away"] = sorted_odds[1].back_price
        
        return result
    
    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            response = self._request("listEventTypes", {"filter": {}})
            return "error" not in response
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Convenience function
def create_client(
    app_key: Optional[str] = None,
    session_token: Optional[str] = None
) -> BetfairClient:
    """Create a Betfair client with optional key override."""
    import os
    key = app_key or os.getenv("BETFAIR_APP_KEY")
    token = session_token or os.getenv("BETFAIR_SESSION_TOKEN")
    
    if not key:
        raise ValueError("Betfair app key required. Set BETFAIR_APP_KEY or pass app_key.")
    
    return BetfairClient(app_key=key, session_token=token)
