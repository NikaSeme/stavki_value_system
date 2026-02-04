"""
Reality Simulator
=================
Simulate real-world trading conditions.
"""

import numpy as np
from typing import Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass
from copy import deepcopy
import logging

if TYPE_CHECKING:
    from ..engine import BacktestConfig

logger = logging.getLogger(__name__)


@dataclass
class RealityScenario:
    """Reality simulation parameters."""
    name: str
    slippage_pct: float
    latency_ms: int
    max_stake: float
    limit_after_bets: Optional[int]  # Get limited after N winning bets
    liquidity_factor: float  # 1.0 = full liquidity, 0.5 = half
    
    def __str__(self):
        return f"Scenario({self.name}): slip={self.slippage_pct:.1%}, lat={self.latency_ms}ms, limit_after={self.limit_after_bets}"


# Predefined scenarios
SCENARIOS = {
    "optimistic": RealityScenario(
        name="optimistic",
        slippage_pct=0.0,
        latency_ms=0,
        max_stake=10000.0,
        limit_after_bets=None,
        liquidity_factor=1.0,
    ),
    "realistic": RealityScenario(
        name="realistic",
        slippage_pct=0.015,  # 1.5% average slippage
        latency_ms=100,
        max_stake=2000.0,
        limit_after_bets=None,
        liquidity_factor=0.9,
    ),
    "pessimistic": RealityScenario(
        name="pessimistic",
        slippage_pct=0.03,  # 3% slippage
        latency_ms=300,
        max_stake=500.0,
        limit_after_bets=100,  # Limited after 100 bets
        liquidity_factor=0.7,
    ),
    "worst_case": RealityScenario(
        name="worst_case",
        slippage_pct=0.05,  # 5% slippage
        latency_ms=1000,
        max_stake=200.0,
        limit_after_bets=30,
        liquidity_factor=0.5,
    ),
}


class RealitySimulator:
    """
    Simulates real-world trading conditions.
    
    Factors modeled:
    - Slippage: Odds move against you before execution
    - Latency: Delay between decision and execution
    - Liquidity: Limited stake acceptance
    - Limits: Bookmaker account restrictions
    """
    
    def __init__(self, scenario: str = "realistic"):
        if scenario in SCENARIOS:
            self.scenario = SCENARIOS[scenario]
        else:
            self.scenario = SCENARIOS["realistic"]
            logger.warning(f"Unknown scenario '{scenario}', using 'realistic'")
            
        self.bets_placed = 0
        self.is_limited = False
        
        logger.info(f"Reality Simulator: {self.scenario}")
    
    def adjust_config(self, config: "BacktestConfig") -> "BacktestConfig":
        """Create adjusted config based on scenario."""
        adjusted = deepcopy(config)
        adjusted.slippage_pct = self.scenario.slippage_pct
        adjusted.latency_ms = self.scenario.latency_ms
        adjusted.max_stake = self.scenario.max_stake
        return adjusted
    
    def apply_slippage(self, odds: float) -> float:
        """
        Apply slippage to odds.
        
        In reality, odds often move against you between
        decision and execution.
        """
        # Random slippage between 0 and scenario max
        actual_slip = np.random.uniform(0, self.scenario.slippage_pct)
        return odds * (1 - actual_slip)
    
    def apply_latency_drift(self, odds: float) -> float:
        """
        Simulate odds drift during latency period.
        
        The longer the latency, the more the odds can move.
        """
        if self.scenario.latency_ms <= 0:
            return odds
            
        # Odds drift: ~0.1% per 100ms
        drift_factor = self.scenario.latency_ms / 1000 * 0.01
        drift = np.random.normal(0, drift_factor)
        
        return odds * (1 + drift)
    
    def apply_liquidity(self, stake: float) -> float:
        """
        Adjust stake based on market liquidity.
        
        Large stakes may not be fully matched.
        """
        # Probability of full fill decreases with stake size
        if stake < 100:
            fill_rate = 1.0
        elif stake < 500:
            fill_rate = 0.95
        elif stake < 1000:
            fill_rate = 0.85
        else:
            fill_rate = 0.7
            
        fill_rate *= self.scenario.liquidity_factor
        
        # Partial fill
        actual_stake = stake * np.random.uniform(fill_rate, 1.0)
        return min(actual_stake, self.scenario.max_stake)
    
    def check_limits(self) -> bool:
        """
        Check if account is limited.
        
        Returns True if bet can be placed.
        """
        if self.is_limited:
            return False
            
        self.bets_placed += 1
        
        if self.scenario.limit_after_bets:
            if self.bets_placed >= self.scenario.limit_after_bets:
                self.is_limited = True
                logger.warning(f"Account LIMITED after {self.bets_placed} bets!")
                return False
                
        return True
    
    def simulate_execution(
        self,
        odds: float,
        stake: float,
    ) -> Dict:
        """
        Full execution simulation.
        
        Returns:
            Dict with adjusted odds, stake, and execution metadata
        """
        if not self.check_limits():
            return {
                "executed": False,
                "reason": "account_limited",
                "odds": 0,
                "stake": 0,
            }
        
        # Apply all reality factors
        adjusted_odds = self.apply_slippage(odds)
        adjusted_odds = self.apply_latency_drift(adjusted_odds)
        adjusted_stake = self.apply_liquidity(stake)
        
        return {
            "executed": True,
            "original_odds": odds,
            "adjusted_odds": adjusted_odds,
            "slippage": (odds - adjusted_odds) / odds,
            "original_stake": stake,
            "adjusted_stake": adjusted_stake,
            "fill_rate": adjusted_stake / stake if stake > 0 else 0,
        }
    
    def reset(self):
        """Reset simulator state."""
        self.bets_placed = 0
        self.is_limited = False
    
    @staticmethod
    def compare_scenarios(results: Dict[str, "BacktestResult"]) -> Dict:
        """
        Compare backtest results across different scenarios.
        
        Args:
            results: Dict mapping scenario name to BacktestResult
            
        Returns:
            Comparison summary
        """
        comparison = {}
        
        for name, result in results.items():
            comparison[name] = {
                "total_bets": result.total_bets,
                "roi": result.roi,
                "total_profit": result.total_profit,
                "max_drawdown": result.max_drawdown,
            }
        
        # Calculate degradation from optimistic
        if "optimistic" in comparison:
            opt_roi = comparison["optimistic"]["roi"]
            for name in comparison:
                if name != "optimistic":
                    degradation = (opt_roi - comparison[name]["roi"]) / opt_roi if opt_roi > 0 else 0
                    comparison[name]["roi_degradation"] = degradation
        
        return comparison
