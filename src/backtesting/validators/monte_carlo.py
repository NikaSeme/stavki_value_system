"""
Monte Carlo Simulator
=====================
Statistical validation through random simulation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass
import logging

if TYPE_CHECKING:
    from ..engine import BacktestResult, BetResult

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    n_simulations: int
    confidence_level: float
    
    # ROI distribution
    roi_mean: float
    roi_std: float
    roi_ci_lower: float
    roi_ci_upper: float
    
    # Risk metrics
    var_5: float  # Value at Risk 5%
    expected_shortfall: float  # Average loss in worst 5%
    
    # Drawdown distribution
    max_dd_mean: float
    max_dd_95: float  # 95th percentile max drawdown
    
    # Probability metrics
    prob_positive_roi: float
    prob_5pct_roi: float
    prob_10pct_roi: float
    
    # Sharpe distribution
    sharpe_mean: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float


class MonteCarloSimulator:
    """
    Monte Carlo simulation for betting strategy validation.
    
    Methods:
    1. Bootstrap resampling - resample bets with replacement
    2. Variance injection - add noise to probabilities
    3. Sequence shuffling - test for order dependence
    """
    
    def __init__(
        self,
        n_simulations: int = 10000,
        confidence_level: float = 0.95,
        random_seed: Optional[int] = None,
    ):
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        
        if random_seed:
            np.random.seed(random_seed)
            
    def simulate(
        self,
        result: "BacktestResult",
        method: str = "bootstrap",
    ) -> Dict:
        """
        Run Monte Carlo simulation.
        
        Args:
            result: BacktestResult from backtest
            method: 'bootstrap', 'variance', or 'both'
            
        Returns:
            Dict with simulation results and confidence intervals
        """
        if not result.bets:
            logger.warning("No bets to simulate")
            return {}
            
        logger.info(f"Running {self.n_simulations} Monte Carlo simulations ({method})")
        
        # Get bet returns
        returns = np.array([
            b.profit / b.stake if b.stake > 0 else 0 
            for b in result.bets
        ])
        stakes = np.array([b.stake for b in result.bets])
        
        if method == "bootstrap" or method == "both":
            bootstrap_results = self._bootstrap_simulate(returns, stakes)
        else:
            bootstrap_results = None
            
        if method == "variance" or method == "both":
            variance_results = self._variance_simulate(result.bets)
        else:
            variance_results = None
            
        # Combine results
        results = bootstrap_results if bootstrap_results else variance_results
        
        return results
    
    def _bootstrap_simulate(
        self,
        returns: np.ndarray,
        stakes: np.ndarray,
    ) -> Dict:
        """Bootstrap resampling simulation."""
        n_bets = len(returns)
        
        roi_simulations = []
        sharpe_simulations = []
        max_dd_simulations = []
        
        for _ in range(self.n_simulations):
            # Resample with replacement
            indices = np.random.choice(n_bets, size=n_bets, replace=True)
            sample_returns = returns[indices]
            sample_stakes = stakes[indices]
            
            # Calculate ROI
            total_profit = np.sum(sample_returns * sample_stakes)
            total_stake = np.sum(sample_stakes)
            roi = total_profit / total_stake if total_stake > 0 else 0
            roi_simulations.append(roi)
            
            # Calculate Sharpe
            if np.std(sample_returns) > 0:
                sharpe = np.mean(sample_returns) / np.std(sample_returns) * np.sqrt(252)
            else:
                sharpe = 0
            sharpe_simulations.append(sharpe)
            
            # Calculate max drawdown
            cumulative = np.cumsum(sample_returns * sample_stakes)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (peak - cumulative) / np.where(peak > 0, peak, 1)
            max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
            max_dd_simulations.append(max_dd)
        
        roi_arr = np.array(roi_simulations)
        sharpe_arr = np.array(sharpe_simulations)
        dd_arr = np.array(max_dd_simulations)
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        
        return {
            "n_simulations": self.n_simulations,
            "confidence_level": self.confidence_level,
            
            # ROI
            "roi_mean": float(np.mean(roi_arr)),
            "roi_std": float(np.std(roi_arr)),
            "roi_ci_lower": float(np.percentile(roi_arr, alpha/2 * 100)),
            "roi_ci_upper": float(np.percentile(roi_arr, (1 - alpha/2) * 100)),
            "roi_median": float(np.median(roi_arr)),
            
            # Risk
            "var_5": float(np.percentile(roi_arr, 5)),
            "expected_shortfall": float(np.mean(roi_arr[roi_arr <= np.percentile(roi_arr, 5)]) if len(roi_arr[roi_arr <= np.percentile(roi_arr, 5)]) > 0 else np.min(roi_arr)),
            
            # Drawdown
            "max_dd_mean": float(np.mean(dd_arr)),
            "max_dd_95": float(np.percentile(dd_arr, 95)),
            
            # Probability
            "prob_positive_roi": float(np.mean(roi_arr > 0)),
            "prob_5pct_roi": float(np.mean(roi_arr > 0.05)),
            "prob_10pct_roi": float(np.mean(roi_arr > 0.10)),
            
            # Sharpe
            "sharpe_mean": float(np.mean(sharpe_arr)),
            "sharpe_ci_lower": float(np.percentile(sharpe_arr, alpha/2 * 100)),
            "sharpe_ci_upper": float(np.percentile(sharpe_arr, (1 - alpha/2) * 100)),
            
            # Distribution data for plotting
            "roi_distribution": roi_arr.tolist(),
            "sharpe_distribution": sharpe_arr.tolist(),
        }
    
    def _variance_simulate(
        self,
        bets: List["BetResult"],
    ) -> Dict:
        """Simulate with variance injection to test robustness."""
        roi_simulations = []
        
        for _ in range(self.n_simulations):
            sim_profit = 0
            sim_stake = 0
            
            for bet in bets:
                # Add noise to probability
                noise = np.random.normal(0, 0.05)  # 5% std deviation
                adj_prob = np.clip(bet.p_model + noise, 0.01, 0.99)
                
                # Recalculate EV
                adj_ev = adj_prob * bet.odds - 1
                
                # Simulate outcome based on adjusted probability
                won = np.random.random() < adj_prob
                
                if adj_ev > 0:  # Only bet if positive EV after noise
                    if won:
                        profit = bet.stake * (bet.odds - 1)
                    else:
                        profit = -bet.stake
                        
                    sim_profit += profit
                    sim_stake += bet.stake
            
            roi = sim_profit / sim_stake if sim_stake > 0 else 0
            roi_simulations.append(roi)
        
        roi_arr = np.array(roi_simulations)
        alpha = 1 - self.confidence_level
        
        return {
            "n_simulations": self.n_simulations,
            "method": "variance_injection",
            "roi_mean": float(np.mean(roi_arr)),
            "roi_std": float(np.std(roi_arr)),
            "roi_ci_lower": float(np.percentile(roi_arr, alpha/2 * 100)),
            "roi_ci_upper": float(np.percentile(roi_arr, (1 - alpha/2) * 100)),
            "prob_positive_roi": float(np.mean(roi_arr > 0)),
        }
    
    def stress_test(
        self,
        result: "BacktestResult",
        scenarios: int = 1000,
    ) -> Dict:
        """
        Run stress test scenarios.
        
        Simulates:
        - Losing streaks
        - Variance spikes
        - Worst case sequences
        """
        if not result.bets:
            return {}
            
        bets = result.bets
        profits = [b.profit for b in bets]
        stakes = [b.stake for b in bets]
        
        # Find worst losing streaks
        worst_streak = 0
        current_streak = 0
        for b in bets:
            if b.profit < 0:
                current_streak += 1
                worst_streak = max(worst_streak, current_streak)
            else:
                current_streak = 0
        
        # Simulate extended losing streaks
        extended_streak_impact = []
        for _ in range(scenarios):
            streak_length = np.random.randint(worst_streak, worst_streak + 10)
            avg_loss = np.mean([p for p in profits if p < 0]) if any(p < 0 for p in profits) else -10
            streak_loss = streak_length * avg_loss
            extended_streak_impact.append(streak_loss)
        
        return {
            "worst_observed_streak": worst_streak,
            "avg_loss_per_bet": float(np.mean([p for p in profits if p < 0])) if any(p < 0 for p in profits) else 0,
            "worst_simulated_streak_loss": float(np.min(extended_streak_impact)),
            "95th_percentile_streak_loss": float(np.percentile(extended_streak_impact, 95)),
        }
