"""
Walk-Forward Validation
=======================
Time-series cross-validation for betting strategies.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Callable, TYPE_CHECKING
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging

if TYPE_CHECKING:
    from ..engine import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-Forward Optimization for betting strategies.
    
    How it works:
    1. Train model on window [T0, T1]
    2. Test on window [T1, T2]
    3. Move forward by step size
    4. Repeat
    
    This prevents lookahead bias and tests true out-of-sample performance.
    """
    
    def __init__(
        self,
        train_months: int = 6,
        test_months: int = 2,
        step_months: int = 1,
    ):
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        
    def validate(
        self,
        data: pd.DataFrame,
        engine: "BacktestEngine",
        retrain_func: Optional[Callable] = None,
        n_folds: Optional[int] = None,
    ) -> List["BacktestResult"]:
        """
        Run walk-forward validation.
        
        Args:
            data: Full historical dataset with 'Date' column
            engine: BacktestEngine to use
            retrain_func: Optional function to retrain model on training data
            n_folds: Maximum number of folds (auto-calculated if None)
            
        Returns:
            List of BacktestResult for each fold
        """
        if "Date" not in data.columns:
            raise ValueError("Data must have 'Date' column")
            
        data = data.copy()
        data["Date"] = pd.to_datetime(data["Date"])
        data = data.sort_values("Date")
        
        start_date = data["Date"].min()
        end_date = data["Date"].max()
        
        # Calculate folds
        folds = self._generate_folds(start_date, end_date, n_folds)
        
        logger.info(f"Walk-Forward: {len(folds)} folds")
        logger.info(f"Train: {self.train_months}mo, Test: {self.test_months}mo, Step: {self.step_months}mo")
        
        results = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
            logger.info(f"Fold {i+1}/{len(folds)}: Train [{train_start.date()} to {train_end.date()}], Test [{test_start.date()} to {test_end.date()}]")
            
            # Split data
            train_data = data[(data["Date"] >= train_start) & (data["Date"] < train_end)]
            test_data = data[(data["Date"] >= test_start) & (data["Date"] < test_end)]
            
            if len(train_data) < 50:
                logger.warning(f"Fold {i+1}: Not enough training data ({len(train_data)} matches)")
                continue
                
            if len(test_data) < 10:
                logger.warning(f"Fold {i+1}: Not enough test data ({len(test_data)} matches)")
                continue
            
            # Retrain model if function provided
            if retrain_func:
                try:
                    retrain_func(train_data)
                except Exception as e:
                    logger.error(f"Fold {i+1}: Retrain failed: {e}")
                    continue
            
            # Run backtest on test data
            result = engine.run(test_data)
            result.fold_id = i + 1
            results.append(result)
            
            logger.info(f"Fold {i+1}: {result.total_bets} bets, ROI={result.roi:.2%}")
        
        # Log aggregate results
        if results:
            total_bets = sum(r.total_bets for r in results)
            total_profit = sum(r.total_profit for r in results)
            total_stake = sum(r.total_stake for r in results)
            agg_roi = total_profit / total_stake if total_stake > 0 else 0
            
            logger.info(f"\n=== Walk-Forward Summary ===")
            logger.info(f"Total Folds: {len(results)}")
            logger.info(f"Total Bets: {total_bets}")
            logger.info(f"Aggregate ROI: {agg_roi:.2%}")
            
            # Consistency check
            profitable_folds = sum(1 for r in results if r.roi > 0)
            logger.info(f"Profitable Folds: {profitable_folds}/{len(results)} ({profitable_folds/len(results)*100:.1f}%)")
        
        return results
    
    def _generate_folds(
        self,
        start_date: datetime,
        end_date: datetime,
        n_folds: Optional[int] = None,
    ) -> List[tuple]:
        """Generate walk-forward fold boundaries."""
        folds = []
        
        # First fold starts at beginning
        train_start = start_date
        
        while True:
            train_end = train_start + relativedelta(months=self.train_months)
            test_start = train_end
            test_end = test_start + relativedelta(months=self.test_months)
            
            # Check if we have enough data
            if test_end > end_date:
                break
                
            folds.append((train_start, train_end, test_start, test_end))
            
            # Move forward
            train_start = train_start + relativedelta(months=self.step_months)
            
            # Limit folds if specified
            if n_folds and len(folds) >= n_folds:
                break
        
        return folds
    
    def get_combined_result(self, results: List["BacktestResult"]) -> dict:
        """Combine results from all folds into summary."""
        if not results:
            return {}
            
        all_bets = []
        for r in results:
            all_bets.extend(r.bets)
            
        total_stake = sum(b.stake for b in all_bets)
        total_profit = sum(b.profit for b in all_bets)
        
        roi_values = [r.roi for r in results]
        
        return {
            "n_folds": len(results),
            "total_bets": len(all_bets),
            "total_stake": total_stake,
            "total_profit": total_profit,
            "aggregate_roi": total_profit / total_stake if total_stake > 0 else 0,
            "avg_roi_per_fold": np.mean(roi_values),
            "std_roi_per_fold": np.std(roi_values),
            "min_roi": np.min(roi_values),
            "max_roi": np.max(roi_values),
            "profitable_folds": sum(1 for r in roi_values if r > 0),
            "consistency": sum(1 for r in roi_values if r > 0) / len(roi_values),
        }
