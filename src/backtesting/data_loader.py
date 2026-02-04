"""
Backtest Data Loader
====================
Unified data loading for backtesting with multiple data sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)


class BacktestDataLoader:
    """
    Load and prepare data for backtesting.
    
    Supports:
    - Historical match data (football-data.co.uk format)
    - Processed feature files
    - Odds snapshots
    - Closing odds for CLV
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir = self.data_dir / "raw"
        self.odds_dir = self.data_dir / "odds"
        
        # Cache
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl: float = 300  # 5 min TTL
        
    def load_historical(
        self,
        leagues: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load historical match data.
        
        Args:
            leagues: List of leagues to load (e.g., ['EPL', 'LaLiga'])
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            DataFrame with historical matches
        """
        # Check cache first
        cache_key = f"hist_{leagues}_{start_date}_{end_date}"
        if cache_key in self._cache:
            import time
            if time.time() - self._cache_timestamps.get(cache_key, 0) < self._cache_ttl:
                logger.info(f"Using cached data ({len(self._cache[cache_key])} matches)")
                return self._cache[cache_key].copy()
        
        # Try Parquet first (5-10x faster than CSV)
        master_parquet = self.processed_dir / "multi_league_features_peopled.parquet"
        master_csv = self.processed_dir / "multi_league_features_peopled.csv"
        
        if master_parquet.exists():
            logger.info(f"Loading Parquet (fast): {master_parquet}")
            df = pd.read_parquet(master_parquet)
        elif master_csv.exists():
            logger.info(f"Loading CSV: {master_csv}")
            df = pd.read_csv(master_csv)
        else:
            # Fallback to individual league files
            df = self._load_league_files(leagues)
            
        if df.empty:
            logger.warning("No data loaded!")
            return df
            
        # Parse dates
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        elif "date" in df.columns:
            df["Date"] = pd.to_datetime(df["date"], errors="coerce")
            
        # Apply date filters
        if start_date and "Date" in df.columns:
            df = df[df["Date"] >= start_date]
        if end_date and "Date" in df.columns:
            df = df[df["Date"] <= end_date]
            
        # Apply league filter
        if leagues:
            league_col = "League" if "League" in df.columns else "league"
            if league_col in df.columns:
                # Normalize league names
                league_map = {
                    "EPL": ["EPL", "E0", "Premier League", "england_premier_league"],
                    "LaLiga": ["LaLiga", "SP1", "La Liga", "spain_la_liga"],
                    "Bundesliga": ["Bundesliga", "D1", "germany_bundesliga"],
                    "SerieA": ["SerieA", "I1", "Serie A", "italy_serie_a"],
                    "Ligue1": ["Ligue1", "F1", "Ligue 1", "france_ligue_1"],
                    "Championship": ["Championship", "E1", "england_championship"],
                }
                
                allowed = []
                for league in leagues:
                    allowed.extend(league_map.get(league, [league]))
                    
                df = df[df[league_col].isin(allowed)]
        
        df = df.sort_values("Date") if "Date" in df.columns else df
        
        # Cache result
        import time
        self._cache[cache_key] = df.copy()
        self._cache_timestamps[cache_key] = time.time()
        
        logger.info(f"Loaded {len(df)} matches")
        return df
    
    def _load_league_files(self, leagues: Optional[List[str]] = None) -> pd.DataFrame:
        """Load individual league CSV files."""
        dfs = []
        
        league_files = {
            "EPL": "epl_historical_2021_2024.csv",
            "LaLiga": "laliga_historical_2021_2024.csv",
            "Bundesliga": "bundesliga_historical_2021_2024.csv",
            "SerieA": "seriea_historical_2021_2024.csv",
            "Ligue1": "ligue1_historical_2021_2024.csv",
            "Championship": "championship_historical_2021_2024.csv",
        }
        
        for league, filename in league_files.items():
            if leagues and league not in leagues:
                continue
                
            filepath = self.processed_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                df["League"] = league
                dfs.append(df)
                
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    def load_features(
        self,
        version: str = "peopled",
    ) -> pd.DataFrame:
        """
        Load pre-computed features.
        
        Args:
            version: Feature version ('peopled', 'v2', 'clean')
            
        Returns:
            DataFrame with features
        """
        version_map = {
            "peopled": "multi_league_features_peopled.csv",
            "v2": "ml_dataset_v2.csv",
            "clean": "multi_league_clean_2021_2024.csv",
            "6leagues": "multi_league_features_6leagues_full.csv",
            "master": "multi_league_master_peopled.csv",
        }
        
        filename = version_map.get(version, version)
        filepath = self.processed_dir / filename
        
        if not filepath.exists():
            filepath = self.data_dir / filename
            
        if filepath.exists():
            logger.info(f"Loading features: {filepath}")
            return pd.read_csv(filepath)
        else:
            logger.error(f"Feature file not found: {filepath}")
            return pd.DataFrame()
    
    def load_odds_snapshots(
        self,
        event_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load odds snapshots for CLV analysis.
        
        Args:
            event_ids: Filter to specific events
            
        Returns:
            DataFrame with odds snapshots over time
        """
        snapshots_dir = self.odds_dir / "snapshots"
        
        if not snapshots_dir.exists():
            logger.warning(f"Snapshots directory not found: {snapshots_dir}")
            return pd.DataFrame()
            
        dfs = []
        for file in snapshots_dir.glob("*.json"):
            with open(file) as f:
                data = json.load(f)
                dfs.append(pd.DataFrame(data))
                
        if not dfs:
            return pd.DataFrame()
            
        df = pd.concat(dfs, ignore_index=True)
        
        if event_ids:
            df = df[df["event_id"].isin(event_ids)]
            
        return df
    
    def get_closing_odds(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Get closing odds for matches.
        
        Uses the latest snapshot before kickoff as closing odds.
        
        Args:
            data: Match data with event_id and kickoff time
            
        Returns:
            DataFrame with closing odds per event
        """
        # Try to get from snapshots
        snapshots = self.load_odds_snapshots()
        
        if snapshots.empty:
            logger.warning("No snapshots available for closing odds")
            # Fall back to using existing odds columns as proxy
            return self._proxy_closing_odds(data)
            
        closing = []
        
        for _, row in data.iterrows():
            event_id = row.get("event_id")
            kickoff = pd.to_datetime(row.get("commence_time", row.get("Date")))
            
            # Get last snapshot before kickoff
            event_snaps = snapshots[snapshots["event_id"] == event_id]
            if event_snaps.empty:
                continue
                
            event_snaps = event_snaps[pd.to_datetime(event_snaps["timestamp"]) < kickoff]
            if event_snaps.empty:
                continue
                
            last_snap = event_snaps.iloc[-1]
            
            closing.append({
                "event_id": event_id,
                "close_h": last_snap.get("odds_home", last_snap.get("home")),
                "close_d": last_snap.get("odds_draw", last_snap.get("draw")),
                "close_a": last_snap.get("odds_away", last_snap.get("away")),
            })
            
        return pd.DataFrame(closing)
    
    def _proxy_closing_odds(self, data: pd.DataFrame) -> pd.DataFrame:
        """Use existing odds as closing odds proxy."""
        result = []
        
        for _, row in data.iterrows():
            event_id = row.get("event_id", row.name)
            
            # Use Pinnacle if available, else Bet365, else average
            h = row.get("PSH", row.get("B365H", row.get("odds_home", 2.0)))
            d = row.get("PSD", row.get("B365D", row.get("odds_draw", 3.5)))
            a = row.get("PSA", row.get("B365A", row.get("odds_away", 3.0)))
            
            result.append({
                "event_id": event_id,
                "close_h": h,
                "close_d": d,
                "close_a": a,
            })
            
        return pd.DataFrame(result)
    
    def get_train_test_split(
        self,
        data: pd.DataFrame,
        train_end: str,
        test_start: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by date for walk-forward.
        
        Args:
            data: Full dataset
            train_end: End date for training (YYYY-MM-DD)
            test_start: Start date for testing (defaults to train_end)
            
        Returns:
            (train_df, test_df)
        """
        if "Date" not in data.columns:
            raise ValueError("Data must have 'Date' column for temporal split")
            
        data["Date"] = pd.to_datetime(data["Date"])
        train_end = pd.to_datetime(train_end)
        test_start = pd.to_datetime(test_start) if test_start else train_end
        
        train = data[data["Date"] < train_end]
        test = data[data["Date"] >= test_start]
        
        return train, test


# Convenience function
def load_backtest_data(
    leagues: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Quick load function for backtest data."""
    loader = BacktestDataLoader()
    return loader.load_historical(leagues, start_date, end_date)
