"""
Bayesian Optimizer
==================
Hyperparameter optimization for betting strategies.
"""

import numpy as np
from typing import Dict, List, Callable, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass
import json
from pathlib import Path
import logging

if TYPE_CHECKING:
    from ..engine import BacktestEngine, BacktestConfig

logger = logging.getLogger(__name__)


@dataclass
class ParameterSpace:
    """Define search space for a parameter."""
    name: str
    min_val: float
    max_val: float
    step: float = 0.01
    param_type: str = "float"  # "float" or "int"


@dataclass
class OptimizationResult:
    """Result of optimization."""
    best_params: Dict[str, float]
    best_score: float
    all_trials: List[Dict]
    param_importance: Dict[str, float]


class BayesianOptimizer:
    """
    Bayesian optimization for betting strategy parameters.
    
    Uses a simple surrogate model approach to efficiently
    explore the parameter space.
    """
    
    # Default parameter spaces
    DEFAULT_SPACES = [
        ParameterSpace("min_ev", 0.02, 0.15, 0.005),
        ParameterSpace("min_odds", 1.2, 2.5, 0.05),
        ParameterSpace("max_odds", 3.0, 15.0, 0.5),
        ParameterSpace("kelly_fraction", 0.05, 0.5, 0.01),
        ParameterSpace("poisson_weight", 0.0, 1.0, 0.05),
        ParameterSpace("catboost_weight", 0.0, 1.0, 0.05),
        ParameterSpace("neural_weight", 0.0, 1.0, 0.05),
    ]
    
    def __init__(
        self,
        objective: str = "roi",  # "roi", "sharpe", "calmar"
        n_iterations: int = 100,
        n_random_starts: int = 20,
        param_spaces: Optional[List[ParameterSpace]] = None,
    ):
        self.objective = objective
        self.n_iterations = n_iterations
        self.n_random_starts = n_random_starts
        self.param_spaces = param_spaces or self.DEFAULT_SPACES
        
        self.trials: List[Dict] = []
        self.best_params: Dict[str, float] = {}
        self.best_score: float = float("-inf")
        
    def optimize(
        self,
        engine: "BacktestEngine",
        data,
        constraint_func: Optional[Callable] = None,
    ) -> OptimizationResult:
        """
        Run optimization.
        
        Args:
            engine: BacktestEngine to use
            data: Data for backtesting
            constraint_func: Optional function to validate params (e.g., sum to 1)
            
        Returns:
            OptimizationResult with best parameters
        """
        logger.info(f"Starting optimization: {self.n_iterations} iterations, objective={self.objective}")
        
        # Random exploration phase
        for i in range(self.n_random_starts):
            params = self._random_params()
            
            if constraint_func and not constraint_func(params):
                continue
                
            score = self._evaluate(engine, data, params)
            self._record_trial(params, score)
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                logger.info(f"New best: {self.objective}={score:.4f}, params={params}")
        
        # Exploitation phase with surrogate model
        for i in range(self.n_random_starts, self.n_iterations):
            # Use simple acquisition function
            params = self._acquisition_sample()
            
            if constraint_func and not constraint_func(params):
                continue
                
            score = self._evaluate(engine, data, params)
            self._record_trial(params, score)
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                logger.info(f"Iteration {i}: New best {self.objective}={score:.4f}")
        
        # Calculate parameter importance
        importance = self._calculate_importance()
        
        logger.info(f"\nOptimization complete!")
        logger.info(f"Best {self.objective}: {self.best_score:.4f}")
        logger.info(f"Best params: {self.best_params}")
        
        return OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            all_trials=self.trials,
            param_importance=importance,
        )
    
    def _random_params(self) -> Dict[str, float]:
        """Generate random parameters within spaces."""
        params = {}
        for space in self.param_spaces:
            n_steps = int((space.max_val - space.min_val) / space.step) + 1
            value = space.min_val + np.random.randint(0, n_steps) * space.step
            params[space.name] = round(value, 4)
        return params
    
    def _acquisition_sample(self) -> Dict[str, float]:
        """Sample using acquisition function (UCB-like)."""
        # Simple approach: perturb best params with decreasing variance
        params = {}
        exploration = max(0.1, 1.0 - len(self.trials) / self.n_iterations)
        
        for space in self.param_spaces:
            if space.name in self.best_params:
                # Perturb around best
                current = self.best_params[space.name]
                range_size = space.max_val - space.min_val
                noise = np.random.normal(0, range_size * exploration * 0.2)
                value = np.clip(current + noise, space.min_val, space.max_val)
                # Snap to grid
                value = space.min_val + round((value - space.min_val) / space.step) * space.step
            else:
                # Random if no best yet
                value = np.random.uniform(space.min_val, space.max_val)
                value = space.min_val + round((value - space.min_val) / space.step) * space.step
                
            params[space.name] = round(value, 4)
            
        return params
    
    def _evaluate(
        self,
        engine: "BacktestEngine",
        data,
        params: Dict[str, float],
    ) -> float:
        """Evaluate parameters."""
        # Update config
        for key, value in params.items():
            if hasattr(engine.config, key):
                setattr(engine.config, key, value)
        
        # Normalize weights if present
        weight_keys = ["poisson_weight", "catboost_weight", "neural_weight"]
        weights = {k: params.get(k, 0) for k in weight_keys if k in params}
        if weights:
            total = sum(weights.values())
            if total > 0:
                for k in weights:
                    setattr(engine.config, k, weights[k] / total)
        
        # Run backtest
        try:
            result = engine.run(data)
            
            if self.objective == "roi":
                return result.roi
            elif self.objective == "sharpe":
                return result.sharpe_ratio
            elif self.objective == "calmar":
                return result.calmar_ratio
            elif self.objective == "profit":
                return result.total_profit
            else:
                return result.roi
                
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return float("-inf")
    
    def _record_trial(self, params: Dict[str, float], score: float):
        """Record trial result."""
        self.trials.append({
            "params": params.copy(),
            "score": score,
            "iteration": len(self.trials),
        })
    
    def _calculate_importance(self) -> Dict[str, float]:
        """Estimate parameter importance from trials."""
        if len(self.trials) < 10:
            return {}
            
        importance = {}
        scores = np.array([t["score"] for t in self.trials])
        
        for space in self.param_spaces:
            values = np.array([t["params"].get(space.name, 0) for t in self.trials])
            
            if np.std(values) > 0:
                # Correlation with score
                corr = np.corrcoef(values, scores)[0, 1]
                importance[space.name] = abs(float(corr)) if not np.isnan(corr) else 0
            else:
                importance[space.name] = 0
                
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
            
        return importance
    
    def save_results(self, path: str = "models/optimization_results.json"):
        """Save optimization results."""
        results = {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "objective": self.objective,
            "n_trials": len(self.trials),
            "trials": self.trials[-10:],  # Last 10 trials
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {path}")


def optimize_per_league(
    engine: "BacktestEngine",
    data,
    leagues: List[str],
    n_iterations: int = 50,
) -> Dict[str, Dict[str, float]]:
    """
    Optimize parameters separately for each league.
    
    Args:
        engine: BacktestEngine
        data: Data with 'League' column
        leagues: List of leagues to optimize
        n_iterations: Iterations per league
        
    Returns:
        Dict mapping league to optimal parameters
    """
    league_params = {}
    
    league_col = "League" if "League" in data.columns else "league"
    
    for league in leagues:
        logger.info(f"\n=== Optimizing {league} ===")
        
        league_data = data[data[league_col] == league]
        
        if len(league_data) < 100:
            logger.warning(f"{league}: Not enough data ({len(league_data)} matches)")
            continue
        
        optimizer = BayesianOptimizer(
            objective="roi",
            n_iterations=n_iterations,
            n_random_starts=min(10, n_iterations // 3),
        )
        
        result = optimizer.optimize(engine, league_data)
        league_params[league] = result.best_params
        
        logger.info(f"{league}: Best ROI={result.best_score:.2%}")
    
    # Save league-specific config
    config_path = Path("models/league_optimized_params.json")
    with open(config_path, "w") as f:
        json.dump(league_params, f, indent=2)
    logger.info(f"\nSaved league params to {config_path}")
    
    return league_params
