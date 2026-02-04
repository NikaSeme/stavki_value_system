"""
Metrics Calculator
==================
Comprehensive betting performance metrics.
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
class MetricsSummary:
    """All metrics in one place."""
    
    # Core
    total_bets: int
    total_stake: float
    total_profit: float
    roi: float
    
    # Win/Loss
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Odds
    avg_odds: float
    avg_ev: float
    
    # Risk
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    var_5: float
    
    # CLV
    clv_avg: float
    clv_hit_rate: float
    
    # Per-league
    by_league: Dict[str, Dict]


class MetricsCalculator:
    """
    Calculate all betting performance metrics.
    
    Metrics included:
    - ROI, Win Rate
    - Sharpe, Sortino, Calmar ratios
    - Max Drawdown, VaR
    - CLV (Closing Line Value)
    - Profit Factor
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        self.risk_free_rate = risk_free_rate
        
    def calculate_all(
        self,
        result: "BacktestResult",
    ) -> MetricsSummary:
        """Calculate all metrics from backtest result."""
        bets = result.bets
        
        if not bets:
            return self._empty_summary()
        
        # Core metrics
        total_bets = len(bets)
        total_stake = sum(b.stake for b in bets)
        total_profit = sum(b.profit for b in bets)
        roi = total_profit / total_stake if total_stake > 0 else 0
        
        # Win/Loss
        wins = [b for b in bets if b.won]
        losses = [b for b in bets if not b.won]
        win_rate = len(wins) / total_bets if total_bets > 0 else 0
        avg_win = np.mean([b.profit for b in wins]) if wins else 0
        avg_loss = np.mean([abs(b.profit) for b in losses]) if losses else 0
        
        gross_profit = sum(b.profit for b in wins) if wins else 0
        gross_loss = sum(abs(b.profit) for b in losses) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Odds and EV
        avg_odds = np.mean([b.odds for b in bets])
        avg_ev = np.mean([b.ev for b in bets])
        
        # Risk metrics
        returns = [b.profit / b.stake if b.stake > 0 else 0 for b in bets]
        max_dd = self._calculate_max_drawdown(bets)
        sharpe = self._calculate_sharpe(returns)
        sortino = self._calculate_sortino(returns)
        calmar = roi / max_dd if max_dd > 0 else 0
        var_5 = self._calculate_var(returns, 0.05)
        
        # CLV
        clv_values = [b.clv for b in bets if b.clv is not None]
        clv_avg = np.mean(clv_values) if clv_values else 0
        clv_hit_rate = sum(1 for c in clv_values if c > 0) / len(clv_values) if clv_values else 0
        
        # Per-league
        by_league = self._calculate_by_league(bets)
        
        return MetricsSummary(
            total_bets=total_bets,
            total_stake=total_stake,
            total_profit=total_profit,
            roi=roi,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_odds=avg_odds,
            avg_ev=avg_ev,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            var_5=var_5,
            clv_avg=clv_avg,
            clv_hit_rate=clv_hit_rate,
            by_league=by_league,
        )
    
    def _calculate_max_drawdown(self, bets: List["BetResult"]) -> float:
        """Calculate maximum drawdown."""
        if not bets:
            return 0.0
            
        cumulative = np.cumsum([b.profit for b in bets])
        peak = np.maximum.accumulate(cumulative)
        
        # Avoid division by zero
        peak_safe = np.where(peak > 0, peak, 1)
        drawdown = (peak - cumulative) / peak_safe
        
        return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
    
    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio (annualized)."""
        if len(returns) < 2:
            return 0.0
            
        mean_return = np.mean(returns) - self.risk_free_rate
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
            
        # Annualize (assuming ~250 bets per year average)
        return float(mean_return / std_return * np.sqrt(252))
    
    def _calculate_sortino(self, returns: List[float]) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        if len(returns) < 2:
            return 0.0
            
        mean_return = np.mean(returns) - self.risk_free_rate
        negative_returns = [r for r in returns if r < 0]
        
        if not negative_returns:
            return float("inf") if mean_return > 0 else 0.0
            
        downside_std = np.std(negative_returns)
        
        if downside_std == 0:
            return 0.0
            
        return float(mean_return / downside_std * np.sqrt(252))
    
    def _calculate_var(self, returns: List[float], alpha: float = 0.05) -> float:
        """Calculate Value at Risk at alpha percentile."""
        if not returns:
            return 0.0
        return float(np.percentile(returns, alpha * 100))
    
    def _calculate_by_league(self, bets: List["BetResult"]) -> Dict[str, Dict]:
        """Calculate metrics per league."""
        leagues = set(b.league for b in bets)
        result = {}
        
        for league in leagues:
            league_bets = [b for b in bets if b.league == league]
            stake = sum(b.stake for b in league_bets)
            profit = sum(b.profit for b in league_bets)
            wins = sum(1 for b in league_bets if b.won)
            
            result[league] = {
                "bets": len(league_bets),
                "stake": round(stake, 2),
                "profit": round(profit, 2),
                "roi": round(profit / stake * 100, 2) if stake > 0 else 0,
                "win_rate": round(wins / len(league_bets) * 100, 2) if league_bets else 0,
            }
            
        return result
    
    def _empty_summary(self) -> MetricsSummary:
        """Return empty summary."""
        return MetricsSummary(
            total_bets=0,
            total_stake=0,
            total_profit=0,
            roi=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            avg_odds=0,
            avg_ev=0,
            max_drawdown=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            var_5=0,
            clv_avg=0,
            clv_hit_rate=0,
            by_league={},
        )
    
    def to_dict(self, summary: MetricsSummary) -> Dict:
        """Convert summary to dictionary."""
        return {
            "core": {
                "total_bets": summary.total_bets,
                "total_stake": round(summary.total_stake, 2),
                "total_profit": round(summary.total_profit, 2),
                "roi_pct": round(summary.roi * 100, 2),
            },
            "performance": {
                "win_rate_pct": round(summary.win_rate * 100, 2),
                "avg_win": round(summary.avg_win, 2),
                "avg_loss": round(summary.avg_loss, 2),
                "profit_factor": round(summary.profit_factor, 2),
                "avg_odds": round(summary.avg_odds, 2),
                "avg_ev_pct": round(summary.avg_ev * 100, 2),
            },
            "risk": {
                "max_drawdown_pct": round(summary.max_drawdown * 100, 2),
                "sharpe_ratio": round(summary.sharpe_ratio, 2),
                "sortino_ratio": round(summary.sortino_ratio, 2),
                "calmar_ratio": round(summary.calmar_ratio, 2),
                "var_5_pct": round(summary.var_5 * 100, 2),
            },
            "clv": {
                "avg_clv_pct": round(summary.clv_avg * 100, 2),
                "clv_hit_rate_pct": round(summary.clv_hit_rate * 100, 2),
            },
            "by_league": summary.by_league,
        }
