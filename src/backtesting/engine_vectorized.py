"""
Vectorized Backtest Engine
==========================
High-performance backtest implementation using NumPy vectorization.
Produces IDENTICAL results to the standard engine, just faster.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime
import logging

from .engine import BacktestConfig, BacktestResult, BetResult

logger = logging.getLogger(__name__)


class VectorizedBacktestEngine:
    """
    Vectorized version of BacktestEngine for high-performance backtesting.
    
    Key differences from standard engine:
    - Uses NumPy vectorization instead of Python loops
    - Processes all matches at once
    - 10-50x faster for large datasets
    
    IMPORTANT: Results should be IDENTICAL to standard engine.
    Run verify_consistency() to confirm.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._model = None
        
    def set_model(self, model_func: Callable):
        """Set model function (must accept DataFrame, return probabilities array)."""
        self._model = model_func
        
    def run(
        self,
        data: pd.DataFrame,
        closing_odds: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Run vectorized backtest.
        
        Args:
            data: Historical match data
            closing_odds: Optional closing odds for CLV
            
        Returns:
            BacktestResult identical to standard engine output
        """
        n = len(data)
        logger.info(f"Running vectorized backtest on {n} matches")
        
        # Extract odds arrays with proper NaN handling
        odds_h = self._safe_odds(data, ["odds_home", "B365H"], 2.0)
        odds_d = self._safe_odds(data, ["odds_draw", "B365D"], 3.5)
        odds_a = self._safe_odds(data, ["odds_away", "B365A"], 3.0)
        
        # Get probabilities
        if self._model:
            probs = self._model(data)
            p_h = probs["home"] if isinstance(probs, dict) else probs[:, 0]
            p_d = probs["draw"] if isinstance(probs, dict) else probs[:, 1]
            p_a = probs["away"] if isinstance(probs, dict) else probs[:, 2]
        else:
            # Default: remove vig from odds
            total = (1/odds_h) + (1/odds_d) + (1/odds_a)
            p_h = (1/odds_h) / total
            p_d = (1/odds_d) / total
            p_a = (1/odds_a) / total
        
        # Calculate EV for each outcome
        ev_h = (p_h * odds_h) - 1
        ev_d = (p_d * odds_d) - 1
        ev_a = (p_a * odds_a) - 1
        
        # Get actual results
        results = data.get("FTR", data.get("result", pd.Series([""] * n))).values
        
        # Extract metadata
        event_ids = data.get("event_id", pd.Series(range(n))).astype(str).values
        dates = pd.to_datetime(data.get("Date", data.get("date", datetime.now()))).values
        leagues = data.get("League", data.get("league", pd.Series(["Unknown"] * n))).values
        home_teams = data.get("HomeTeam", data.get("home_team", pd.Series([""] * n))).values
        away_teams = data.get("AwayTeam", data.get("away_team", pd.Series([""] * n))).values
        
        bets = []
        
        # Process each outcome type
        for outcome_code, probs, odds, ev in [
            ("H", p_h, odds_h, ev_h),
            ("D", p_d, odds_d, ev_d),
            ("A", p_a, odds_a, ev_a),
        ]:
            # Find qualifying bets (vectorized)
            qualifies = (
                (ev >= self.config.min_ev) & 
                (odds >= self.config.min_odds) & 
                (odds <= self.config.max_odds)
            )
            
            # Calculate stakes (vectorized)
            edge = probs - (1 / odds)
            kelly_full = np.where(odds > 1, edge / (odds - 1), 0)
            stakes = np.maximum(0, kelly_full * self.config.kelly_fraction * 100)
            stakes = np.minimum(stakes, self.config.max_stake)
            
            # Only keep bets with positive stake
            qualifies = qualifies & (stakes > 0)
            
            # Apply slippage
            adjusted_odds = odds * (1 - self.config.slippage_pct)
            
            # Calculate profit
            won = (results == outcome_code)
            profit = np.where(won, stakes * (adjusted_odds - 1), -stakes)
            
            # Get indices of qualifying bets
            indices = np.where(qualifies)[0]
            
            # Create bet results
            for i in indices:
                bet = BetResult(
                    event_id=str(event_ids[i]),
                    date=pd.Timestamp(dates[i]).to_pydatetime(),
                    league=str(leagues[i]),
                    home_team=str(home_teams[i]),
                    away_team=str(away_teams[i]),
                    outcome=outcome_code,
                    odds=float(odds[i]),
                    stake=float(stakes[i]),
                    p_model=float(probs[i]),
                    ev=float(ev[i]),
                    won=bool(won[i]),
                    profit=float(profit[i]),
                    clv=None,  # TODO: add CLV if closing_odds provided
                )
                bets.append(bet)
        
        # Sort bets by date
        bets.sort(key=lambda b: b.date)
        
        # Create result
        result = BacktestResult(
            bets=bets,
            config=self.config,
            start_date=pd.Timestamp(dates.min()).to_pydatetime() if len(dates) > 0 else datetime.now(),
            end_date=pd.Timestamp(dates.max()).to_pydatetime() if len(dates) > 0 else datetime.now(),
        )
        result.compute_metrics()
        
        logger.info(f"Vectorized backtest complete: {result.total_bets} bets, ROI={result.roi:.2%}")
        
        return result
    
    def _safe_odds(
        self,
        data: pd.DataFrame,
        columns: List[str],
        default: float,
    ) -> np.ndarray:
        """Extract odds with NaN handling."""
        for col in columns:
            if col in data.columns:
                values = data[col].values.copy()
                # Replace NaN and invalid values
                invalid = np.isnan(values) | (values <= 1)
                values[invalid] = default
                return values
        return np.full(len(data), default)


def verify_consistency(
    data: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
    tolerance: float = 0.0001,
) -> Dict:
    """
    Verify that vectorized engine produces identical results to standard.
    
    Args:
        data: Test data
        config: Optional config
        tolerance: Numerical tolerance for comparison
        
    Returns:
        Dict with comparison results
    """
    from .engine import BacktestEngine
    
    config = config or BacktestConfig()
    
    # Run both engines
    standard = BacktestEngine(config=config)
    vectorized = VectorizedBacktestEngine(config=config)
    
    result_std = standard.run(data)
    result_vec = vectorized.run(data)
    
    # Compare key metrics
    checks = {
        "total_bets_match": result_std.total_bets == result_vec.total_bets,
        "total_stake_match": abs(result_std.total_stake - result_vec.total_stake) < tolerance,
        "total_profit_match": abs(result_std.total_profit - result_vec.total_profit) < tolerance,
        "roi_match": abs(result_std.roi - result_vec.roi) < tolerance,
    }
    
    all_passed = all(checks.values())
    
    return {
        "passed": all_passed,
        "checks": checks,
        "standard": {
            "bets": result_std.total_bets,
            "stake": result_std.total_stake,
            "profit": result_std.total_profit,
            "roi": result_std.roi,
        },
        "vectorized": {
            "bets": result_vec.total_bets,
            "stake": result_vec.total_stake,
            "profit": result_vec.total_profit,
            "roi": result_vec.roi,
        },
    }
