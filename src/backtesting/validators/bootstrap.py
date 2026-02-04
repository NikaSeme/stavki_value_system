"""
Bootstrap Confidence Intervals
==============================
Statistical confidence interval calculation.
"""

import numpy as np
from typing import Dict, List, Tuple, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from ..engine import BacktestResult

logger = logging.getLogger(__name__)


class BootstrapValidator:
    """
    Bootstrap confidence interval calculator.
    
    Provides robust confidence intervals for key metrics
    without assuming normal distribution.
    """
    
    def __init__(
        self,
        n_bootstrap: int = 5000,
        confidence_level: float = 0.95,
        random_seed: int = 42,
    ):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        np.random.seed(random_seed)
        
    def calculate_ci(
        self,
        result: "BacktestResult",
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate confidence intervals for all metrics.
        
        Args:
            result: BacktestResult to analyze
            
        Returns:
            Dict mapping metric name to (lower, upper) CI
        """
        if not result.bets:
            return {}
            
        returns = np.array([b.profit / b.stake if b.stake > 0 else 0 for b in result.bets])
        stakes = np.array([b.stake for b in result.bets])
        won = np.array([1 if b.won else 0 for b in result.bets])
        odds = np.array([b.odds for b in result.bets])
        
        # Bootstrap samples
        n = len(returns)
        
        roi_samples = []
        win_rate_samples = []
        avg_odds_samples = []
        sharpe_samples = []
        
        for _ in range(self.n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            
            # ROI
            sample_profit = np.sum(returns[idx] * stakes[idx])
            sample_stake = np.sum(stakes[idx])
            roi_samples.append(sample_profit / sample_stake if sample_stake > 0 else 0)
            
            # Win rate
            win_rate_samples.append(np.mean(won[idx]))
            
            # Avg odds
            avg_odds_samples.append(np.mean(odds[idx]))
            
            # Sharpe
            sample_returns = returns[idx]
            if np.std(sample_returns) > 0:
                sharpe_samples.append(np.mean(sample_returns) / np.std(sample_returns) * np.sqrt(252))
            else:
                sharpe_samples.append(0)
        
        alpha = 1 - self.confidence_level
        
        return {
            "roi": self._percentile_ci(roi_samples, alpha),
            "win_rate": self._percentile_ci(win_rate_samples, alpha),
            "avg_odds": self._percentile_ci(avg_odds_samples, alpha),
            "sharpe": self._percentile_ci(sharpe_samples, alpha),
        }
    
    def _percentile_ci(
        self,
        samples: List[float],
        alpha: float,
    ) -> Tuple[float, float]:
        """Calculate percentile-based confidence interval."""
        arr = np.array(samples)
        lower = np.percentile(arr, alpha/2 * 100)
        upper = np.percentile(arr, (1 - alpha/2) * 100)
        return (float(lower), float(upper))
    
    def is_significant(
        self,
        result: "BacktestResult",
        baseline_roi: float = 0.0,
    ) -> Dict:
        """
        Test if ROI is significantly different from baseline.
        
        Args:
            result: BacktestResult to test
            baseline_roi: Null hypothesis ROI (default 0)
            
        Returns:
            Dict with significance test results
        """
        ci = self.calculate_ci(result)
        roi_lower, roi_upper = ci.get("roi", (0, 0))
        
        # Check if baseline is outside CI
        is_significant = baseline_roi < roi_lower or baseline_roi > roi_upper
        
        # Calculate p-value estimate
        returns = np.array([b.profit / b.stake if b.stake > 0 else 0 for b in result.bets])
        stakes = np.array([b.stake for b in result.bets])
        
        observed_roi = result.roi
        
        # Bootstrap under null
        null_rois = []
        for _ in range(self.n_bootstrap):
            idx = np.random.choice(len(returns), size=len(returns), replace=True)
            centered = returns[idx] - (np.mean(returns) - baseline_roi)
            roi = np.sum(centered * stakes[idx]) / np.sum(stakes[idx])
            null_rois.append(roi)
        
        # p-value: proportion of null samples >= observed
        p_value = np.mean(np.array(null_rois) >= observed_roi)
        
        return {
            "is_significant": is_significant,
            "confidence_level": self.confidence_level,
            "ci_lower": roi_lower,
            "ci_upper": roi_upper,
            "observed_roi": observed_roi,
            "baseline_roi": baseline_roi,
            "p_value": float(p_value),
        }
