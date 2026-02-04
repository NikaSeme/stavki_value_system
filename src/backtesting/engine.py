"""
Professional Backtest Engine
============================
Core engine for running backtests with multiple validation methods.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest runs."""
    
    # Strategy parameters
    min_ev: float = 0.05
    min_odds: float = 1.30
    max_odds: float = 10.0
    kelly_fraction: float = 0.20
    
    # Model weights
    poisson_weight: float = 0.35
    catboost_weight: float = 0.40
    neural_weight: float = 0.25
    
    # Walk-forward settings
    train_months: int = 6
    test_months: int = 2
    step_months: int = 1
    
    # Monte Carlo
    n_simulations: int = 10000
    confidence_level: float = 0.95
    
    # Reality simulation
    slippage_pct: float = 0.02
    latency_ms: int = 100
    max_stake: float = 1000.0
    
    # Filters
    leagues: List[str] = field(default_factory=lambda: [
        "EPL", "LaLiga", "Bundesliga", "SerieA", "Ligue1", "Championship"
    ])
    
    def to_dict(self) -> Dict:
        return {
            "min_ev": self.min_ev,
            "min_odds": self.min_odds,
            "max_odds": self.max_odds,
            "kelly_fraction": self.kelly_fraction,
            "poisson_weight": self.poisson_weight,
            "catboost_weight": self.catboost_weight,
            "neural_weight": self.neural_weight,
            "train_months": self.train_months,
            "test_months": self.test_months,
            "n_simulations": self.n_simulations,
            "slippage_pct": self.slippage_pct,
            "leagues": self.leagues,
        }


@dataclass 
class BetResult:
    """Single bet result."""
    event_id: str
    date: datetime
    league: str
    home_team: str
    away_team: str
    outcome: str  # 'H', 'D', 'A'
    odds: float
    stake: float
    p_model: float
    ev: float
    won: bool
    profit: float
    clv: Optional[float] = None  # Closing Line Value
    

@dataclass
class BacktestResult:
    """Complete backtest results."""
    
    bets: List[BetResult]
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    
    # Computed metrics
    total_bets: int = 0
    total_stake: float = 0.0
    total_profit: float = 0.0
    roi: float = 0.0
    win_rate: float = 0.0
    avg_odds: float = 0.0
    avg_ev: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    clv_avg: float = 0.0
    
    # Confidence intervals
    roi_ci_lower: float = 0.0
    roi_ci_upper: float = 0.0
    
    # Per-league breakdown
    league_results: Dict[str, Dict] = field(default_factory=dict)
    
    # Walk-forward fold identifier
    fold_id: Optional[int] = None
    
    def compute_metrics(self):
        """Calculate all metrics from bets."""
        if not self.bets:
            return
            
        self.total_bets = len(self.bets)
        self.total_stake = sum(b.stake for b in self.bets)
        self.total_profit = sum(b.profit for b in self.bets)
        
        if self.total_stake > 0:
            self.roi = self.total_profit / self.total_stake
            
        wins = sum(1 for b in self.bets if b.won)
        self.win_rate = wins / self.total_bets if self.total_bets > 0 else 0
        
        self.avg_odds = np.mean([b.odds for b in self.bets])
        self.avg_ev = np.mean([b.ev for b in self.bets])
        
        # CLV
        clv_values = [b.clv for b in self.bets if b.clv is not None]
        if clv_values:
            self.clv_avg = np.mean(clv_values)
        
        # Drawdown calculation
        cumulative = np.cumsum([b.profit for b in self.bets])
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / np.where(peak > 0, peak, 1)
        self.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Sharpe Ratio (annualized)
        returns = [b.profit / b.stake if b.stake > 0 else 0 for b in self.bets]
        if len(returns) > 1 and np.std(returns) > 0:
            self.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # Sortino (downside deviation only)
        negative_returns = [r for r in returns if r < 0]
        if negative_returns and np.std(negative_returns) > 0:
            self.sortino_ratio = np.mean(returns) / np.std(negative_returns) * np.sqrt(252)
        
        # Calmar
        if self.max_drawdown > 0:
            self.calmar_ratio = self.roi / self.max_drawdown
            
        # Per-league
        self._compute_league_breakdown()
        
    def _compute_league_breakdown(self):
        """Compute metrics per league."""
        leagues = set(b.league for b in self.bets)
        
        for league in leagues:
            league_bets = [b for b in self.bets if b.league == league]
            stake = sum(b.stake for b in league_bets)
            profit = sum(b.profit for b in league_bets)
            wins = sum(1 for b in league_bets if b.won)
            
            self.league_results[league] = {
                "bets": len(league_bets),
                "stake": stake,
                "profit": profit,
                "roi": profit / stake if stake > 0 else 0,
                "win_rate": wins / len(league_bets) if league_bets else 0,
                "avg_odds": np.mean([b.odds for b in league_bets]),
            }
    
    def to_dict(self) -> Dict:
        """Export as dictionary."""
        return {
            "summary": {
                "total_bets": self.total_bets,
                "total_stake": round(self.total_stake, 2),
                "total_profit": round(self.total_profit, 2),
                "roi": round(self.roi * 100, 2),
                "win_rate": round(self.win_rate * 100, 2),
                "avg_odds": round(self.avg_odds, 2),
                "avg_ev": round(self.avg_ev * 100, 2),
            },
            "risk": {
                "max_drawdown": round(self.max_drawdown * 100, 2),
                "sharpe_ratio": round(self.sharpe_ratio, 2),
                "sortino_ratio": round(self.sortino_ratio, 2),
                "calmar_ratio": round(self.calmar_ratio, 2),
            },
            "clv": {
                "avg_clv": round(self.clv_avg * 100, 2),
            },
            "confidence": {
                "roi_95_lower": round(self.roi_ci_lower * 100, 2),
                "roi_95_upper": round(self.roi_ci_upper * 100, 2),
            },
            "by_league": self.league_results,
            "config": self.config.to_dict(),
        }


class BacktestEngine:
    """
    Professional Backtesting Engine.
    
    Features:
    - Walk-Forward Optimization
    - Monte Carlo Simulation
    - Reality Simulation (slippage, latency, limits)
    - CLV Tracking
    - Per-league analysis
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.data_loader = None
        self._model = None
        self._strategy = None
        
    def set_model(self, model_func: Callable):
        """Set the prediction model function."""
        self._model = model_func
        
    def set_strategy(self, strategy_func: Callable):
        """Set the betting strategy function."""
        self._strategy = strategy_func
        
    def run(
        self,
        data: pd.DataFrame,
        odds_data: Optional[pd.DataFrame] = None,
        closing_odds: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Run basic backtest.
        
        Args:
            data: Historical match data with features
            odds_data: Odds at bet time
            closing_odds: Closing odds for CLV calculation
            
        Returns:
            BacktestResult with all metrics
        """
        logger.info(f"Running backtest on {len(data)} matches")
        
        bets = []
        
        for idx, row in data.iterrows():
            # Get model prediction
            if self._model:
                probs = self._model(row)
            else:
                probs = self._default_probs(row)
            
            # Get odds with NaN handling
            odds_home = row.get("odds_home", row.get("B365H", 2.0))
            odds_draw = row.get("odds_draw", row.get("B365D", 3.5))
            odds_away = row.get("odds_away", row.get("B365A", 3.0))
            
            # Handle NaN and invalid odds
            if pd.isna(odds_home) or odds_home <= 1:
                odds_home = 2.0
            if pd.isna(odds_draw) or odds_draw <= 1:
                odds_draw = 3.5
            if pd.isna(odds_away) or odds_away <= 1:
                odds_away = 3.0
            
            # Calculate EV for each outcome
            outcomes = [
                ("H", probs["home"], odds_home),
                ("D", probs["draw"], odds_draw),
                ("A", probs["away"], odds_away),
            ]
            
            for outcome, p_model, odds in outcomes:
                ev = (p_model * odds) - 1
                
                # Filter
                if ev < self.config.min_ev:
                    continue
                if odds < self.config.min_odds or odds > self.config.max_odds:
                    continue
                    
                # Kelly stake
                edge = p_model - (1 / odds)
                kelly_full = edge / (odds - 1) if odds > 1 else 0
                stake = max(0, kelly_full * self.config.kelly_fraction * 100)
                
                if stake <= 0:
                    continue
                
                # Apply reality constraints
                stake = min(stake, self.config.max_stake)
                
                # Apply slippage
                adjusted_odds = odds * (1 - self.config.slippage_pct)
                
                # Determine result
                actual_result = row.get("FTR", row.get("result", ""))
                won = (actual_result == outcome)
                
                if won:
                    profit = stake * (adjusted_odds - 1)
                else:
                    profit = -stake
                    
                # CLV
                clv = None
                if closing_odds is not None:
                    close_odds = self._get_closing_odds(row, outcome, closing_odds)
                    if close_odds and close_odds > 0:
                        clv = (odds - close_odds) / close_odds
                
                bet = BetResult(
                    event_id=str(row.get("event_id", idx)),
                    date=pd.to_datetime(row.get("Date", row.get("date", datetime.now()))),
                    league=row.get("League", row.get("league", "Unknown")),
                    home_team=row.get("HomeTeam", row.get("home_team", "")),
                    away_team=row.get("AwayTeam", row.get("away_team", "")),
                    outcome=outcome,
                    odds=odds,
                    stake=stake,
                    p_model=p_model,
                    ev=ev,
                    won=won,
                    profit=profit,
                    clv=clv,
                )
                bets.append(bet)
        
        # Create result
        result = BacktestResult(
            bets=bets,
            config=self.config,
            start_date=data["Date"].min() if "Date" in data.columns else datetime.now(),
            end_date=data["Date"].max() if "Date" in data.columns else datetime.now(),
        )
        result.compute_metrics()
        
        logger.info(f"Backtest complete: {result.total_bets} bets, ROI={result.roi:.2%}")
        
        return result
    
    def run_walk_forward(
        self,
        data: pd.DataFrame,
        retrain_func: Optional[Callable] = None,
        n_folds: int = 12,
    ) -> List[BacktestResult]:
        """
        Run Walk-Forward optimization.
        
        Args:
            data: Full historical data
            retrain_func: Function to retrain model on training set
            n_folds: Number of walk-forward folds
            
        Returns:
            List of BacktestResult for each fold
        """
        from .validators.walk_forward import WalkForwardValidator
        
        validator = WalkForwardValidator(
            train_months=self.config.train_months,
            test_months=self.config.test_months,
            step_months=self.config.step_months,
        )
        
        return validator.validate(
            data=data,
            engine=self,
            retrain_func=retrain_func,
            n_folds=n_folds,
        )
    
    def run_monte_carlo(
        self,
        result: BacktestResult,
        n_simulations: int = 10000,
    ) -> Dict:
        """
        Run Monte Carlo simulation on backtest results.
        
        Args:
            result: BacktestResult to simulate
            n_simulations: Number of random simulations
            
        Returns:
            Dict with confidence intervals and distributions
        """
        from .validators.monte_carlo import MonteCarloSimulator
        
        simulator = MonteCarloSimulator(
            n_simulations=n_simulations,
            confidence_level=self.config.confidence_level,
        )
        
        return simulator.simulate(result)
    
    def run_with_reality(
        self,
        data: pd.DataFrame,
        scenario: str = "realistic",
    ) -> BacktestResult:
        """
        Run backtest with reality simulation.
        
        Args:
            data: Historical data
            scenario: 'optimistic', 'realistic', 'pessimistic', 'worst_case'
            
        Returns:
            BacktestResult with reality adjustments
        """
        from .reality.simulator import RealitySimulator
        
        simulator = RealitySimulator(scenario=scenario)
        adjusted_config = simulator.adjust_config(self.config)
        
        # Run with adjusted config
        original_config = self.config
        self.config = adjusted_config
        result = self.run(data)
        self.config = original_config
        
        return result
    
    def _default_probs(self, row: pd.Series) -> Dict[str, float]:
        """Default probability calculation from odds."""
        h = row.get("odds_home", row.get("B365H", 2.0))
        d = row.get("odds_draw", row.get("B365D", 3.5))
        a = row.get("odds_away", row.get("B365A", 3.0))
        
        # Handle NaN and invalid values
        if pd.isna(h) or h <= 1:
            h = 2.0
        if pd.isna(d) or d <= 1:
            d = 3.5
        if pd.isna(a) or a <= 1:
            a = 3.0
        
        # Remove vig
        total = (1/h) + (1/d) + (1/a)
        
        return {
            "home": (1/h) / total,
            "draw": (1/d) / total,
            "away": (1/a) / total,
        }
    
    def _get_closing_odds(
        self,
        row: pd.Series,
        outcome: str,
        closing_odds: pd.DataFrame,
    ) -> Optional[float]:
        """Get closing odds for CLV calculation."""
        # Try to match by event_id or date+teams
        event_id = row.get("event_id")
        
        if event_id and "event_id" in closing_odds.columns:
            match = closing_odds[closing_odds["event_id"] == event_id]
            if len(match) > 0:
                col = f"close_{outcome.lower()}"
                if col in match.columns:
                    return match[col].iloc[0]
        
        return None


def run_professional_backtest(
    data_path: str = "data/processed/multi_league_features_peopled.csv",
    config: Optional[BacktestConfig] = None,
) -> BacktestResult:
    """
    Run a complete professional backtest.
    
    Args:
        data_path: Path to processed data file
        config: Optional backtest configuration
        
    Returns:
        Complete BacktestResult
    """
    # Load data
    data = pd.read_csv(data_path)
    
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"])
        data = data.sort_values("Date")
    
    # Create engine
    engine = BacktestEngine(config=config)
    
    # Run backtest
    result = engine.run(data)
    
    # Run Monte Carlo
    mc_result = engine.run_monte_carlo(result)
    result.roi_ci_lower = mc_result.get("roi_ci_lower", 0)
    result.roi_ci_upper = mc_result.get("roi_ci_upper", 0)
    
    return result


if __name__ == "__main__":
    # Quick test
    config = BacktestConfig(min_ev=0.05, kelly_fraction=0.2)
    result = run_professional_backtest(config=config)
    
    print("\n=== BACKTEST RESULTS ===")
    print(json.dumps(result.to_dict(), indent=2))
