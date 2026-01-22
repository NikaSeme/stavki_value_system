from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import requests


@dataclass
class OddsAPIConfig:
    api_key: str
    base_url: str = "https://api.the-odds-api.com"
    timeout_s: int = 20
    max_retries: int = 3
    backoff_s: float = 1.5


class OddsAPIError(RuntimeError):
    pass


def _get_env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if not v:
        raise OddsAPIError(f"Missing required env var: {name}")
    return v


def load_config_from_env() -> OddsAPIConfig:
    return OddsAPIConfig(
        api_key=_get_env("ODDS_API_KEY"),
        base_url=os.getenv("ODDS_API_BASE", "https://api.the-odds-api.com"),
    )


def _request_with_retries(url: str, params: Dict[str, Any], cfg: OddsAPIConfig) -> requests.Response:
    last_err: Optional[Exception] = None
    for attempt in range(cfg.max_retries):
        try:
            r = requests.get(url, params=params, timeout=cfg.timeout_s)
            if r.status_code == 429:
                # Rate limited. Back off and retry.
                time.sleep(cfg.backoff_s * (attempt + 1))
                continue
            return r
        except Exception as e:
            last_err = e
            time.sleep(cfg.backoff_s * (attempt + 1))
    raise OddsAPIError(f"Request failed after retries. Last error: {last_err}")


def list_sports(cfg: Optional[OddsAPIConfig] = None, all_sports: bool = False) -> List[Dict[str, Any]]:
    cfg = cfg or load_config_from_env()
    url = f"{cfg.base_url}/v4/sports/"
    params = {"apiKey": cfg.api_key}
    if all_sports:
        params["all"] = "true"
    r = _request_with_retries(url, params, cfg)
    if r.status_code != 200:
        raise OddsAPIError(f"list_sports failed: {r.status_code} {r.text[:200]}")
    return r.json()


def fetch_odds(
    sport_key: str,
    regions: str,
    markets: str,
    *,
    odds_format: str = "decimal",
    date_format: str = "iso",
    bookmakers: Optional[str] = None,
    commence_time_from: Optional[str] = None,
    commence_time_to: Optional[str] = None,
    cfg: Optional[OddsAPIConfig] = None,
) -> List[Dict[str, Any]]:
    """Fetch odds for a sport.

    IMPORTANT: Usage cost is markets x regions (docs). Keep requests tight.
    """
    cfg = cfg or load_config_from_env()
    url = f"{cfg.base_url}/v4/sports/{sport_key}/odds/"
    params: Dict[str, Any] = {
        "apiKey": cfg.api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    if commence_time_from:
        params["commenceTimeFrom"] = commence_time_from
    if commence_time_to:
        params["commenceTimeTo"] = commence_time_to

    r = _request_with_retries(url, params, cfg)
    if r.status_code != 200:
        raise OddsAPIError(f"fetch_odds failed: {r.status_code} {r.text[:200]}")
    return r.json()
