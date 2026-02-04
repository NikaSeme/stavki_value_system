"""
Report Generator
================
Generate professional backtest reports.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from ..engine import BacktestResult
    from ..metrics.calculator import MetricsSummary

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate backtest reports in multiple formats."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate(
        self,
        result: "BacktestResult",
        monte_carlo: Optional[Dict] = None,
        walk_forward: Optional[Dict] = None,
        name: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate full report.
        
        Args:
            result: BacktestResult from backtest
            monte_carlo: Optional Monte Carlo simulation results
            walk_forward: Optional Walk-Forward validation results
            name: Optional report name
            
        Returns:
            Dict with paths to generated files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = name or f"backtest_{timestamp}"
        
        paths = {}
        
        # Generate Markdown
        md_path = self.output_dir / f"{name}.md"
        self._generate_markdown(result, monte_carlo, walk_forward, md_path)
        paths["markdown"] = str(md_path)
        
        # Generate JSON
        json_path = self.output_dir / f"{name}.json"
        self._generate_json(result, monte_carlo, walk_forward, json_path)
        paths["json"] = str(json_path)
        
        logger.info(f"Generated reports: {paths}")
        return paths
    
    def _generate_markdown(
        self,
        result: "BacktestResult",
        monte_carlo: Optional[Dict],
        walk_forward: Optional[Dict],
        path: Path,
    ):
        """Generate Markdown report."""
        md = []
        
        # Header
        md.append(f"# 📊 Backtest Report")
        md.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md.append(f"\n**Period:** {result.start_date.date()} to {result.end_date.date()}")
        md.append("")
        
        # Summary
        md.append("## ✅ Summary")
        md.append("")
        md.append("| Metric | Value |")
        md.append("|--------|------:|")
        md.append(f"| Total Bets | {result.total_bets} |")
        md.append(f"| Total Stake | €{result.total_stake:,.2f} |")
        md.append(f"| Total Profit | €{result.total_profit:,.2f} |")
        md.append(f"| **ROI** | **{result.roi*100:.2f}%** |")
        md.append(f"| Win Rate | {result.win_rate*100:.1f}% |")
        md.append(f"| Avg Odds | {result.avg_odds:.2f} |")
        md.append("")
        
        # Risk Metrics
        md.append("## 📉 Risk Metrics")
        md.append("")
        md.append("| Metric | Value | Target |")
        md.append("|--------|------:|-------:|")
        md.append(f"| Max Drawdown | {result.max_drawdown*100:.1f}% | <25% |")
        md.append(f"| Sharpe Ratio | {result.sharpe_ratio:.2f} | >1.0 |")
        md.append(f"| Sortino Ratio | {result.sortino_ratio:.2f} | >1.5 |")
        md.append(f"| Calmar Ratio | {result.calmar_ratio:.2f} | >0.3 |")
        md.append("")
        
        # CLV
        if result.clv_avg != 0:
            md.append("## 📈 CLV (Closing Line Value)")
            md.append("")
            md.append(f"- **Average CLV:** {result.clv_avg*100:.2f}%")
            md.append(f"- A positive CLV indicates you're beating the market")
            md.append("")
        
        # Monte Carlo
        if monte_carlo:
            md.append("## 🎲 Monte Carlo Simulation")
            md.append(f"\n*{monte_carlo.get('n_simulations', 10000):,} simulations*")
            md.append("")
            md.append("| Metric | Value |")
            md.append("|--------|------:|")
            md.append(f"| 95% CI Lower | {monte_carlo.get('roi_ci_lower', 0)*100:.2f}% |")
            md.append(f"| 95% CI Upper | {monte_carlo.get('roi_ci_upper', 0)*100:.2f}% |")
            md.append(f"| VaR (5%) | {monte_carlo.get('var_5', 0)*100:.2f}% |")
            md.append(f"| Prob. Positive ROI | {monte_carlo.get('prob_positive_roi', 0)*100:.1f}% |")
            md.append(f"| Prob. >5% ROI | {monte_carlo.get('prob_5pct_roi', 0)*100:.1f}% |")
            md.append("")
        
        # Walk-Forward
        if walk_forward:
            md.append("## 🔄 Walk-Forward Validation")
            md.append("")
            md.append("| Metric | Value |")
            md.append("|--------|------:|")
            md.append(f"| Folds | {walk_forward.get('n_folds', 0)} |")
            md.append(f"| Aggregate ROI | {walk_forward.get('aggregate_roi', 0)*100:.2f}% |")
            md.append(f"| Avg ROI per Fold | {walk_forward.get('avg_roi_per_fold', 0)*100:.2f}% |")
            md.append(f"| Consistency | {walk_forward.get('consistency', 0)*100:.1f}% |")
            md.append(f"| Profitable Folds | {walk_forward.get('profitable_folds', 0)}/{walk_forward.get('n_folds', 0)} |")
            md.append("")
        
        # Per-League
        if result.league_results:
            md.append("## 🏆 Per-League Results")
            md.append("")
            md.append("| League | Bets | Profit | ROI | Win Rate |")
            md.append("|--------|-----:|-------:|----:|---------:|")
            for league, stats in sorted(result.league_results.items(), 
                                        key=lambda x: x[1].get("roi", 0), 
                                        reverse=True):
                md.append(f"| {league} | {stats.get('bets', 0)} | €{stats.get('profit', 0):.2f} | {stats.get('roi', 0)*100:.1f}% | {stats.get('win_rate', 0)*100:.1f}% |")
            md.append("")
        
        # Config
        md.append("## ⚙️ Configuration")
        md.append("")
        md.append("```json")
        md.append(json.dumps(result.config.to_dict(), indent=2))
        md.append("```")
        
        # Write file
        with open(path, "w") as f:
            f.write("\n".join(md))
    
    def _generate_json(
        self,
        result: "BacktestResult",
        monte_carlo: Optional[Dict],
        walk_forward: Optional[Dict],
        path: Path,
    ):
        """Generate JSON report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "period": {
                "start": result.start_date.isoformat() if result.start_date else None,
                "end": result.end_date.isoformat() if result.end_date else None,
            },
            "results": result.to_dict(),
        }
        
        if monte_carlo:
            # Remove large distribution arrays for JSON
            mc_summary = {k: v for k, v in monte_carlo.items() 
                        if not k.endswith("_distribution")}
            report["monte_carlo"] = mc_summary
            
        if walk_forward:
            report["walk_forward"] = walk_forward
        
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)


def generate_quick_report(result: "BacktestResult") -> str:
    """Generate a quick console report."""
    lines = [
        "",
        "=" * 50,
        "BACKTEST RESULTS",
        "=" * 50,
        f"Total Bets:    {result.total_bets}",
        f"Total Stake:   €{result.total_stake:,.2f}",
        f"Total Profit:  €{result.total_profit:,.2f}",
        f"ROI:           {result.roi*100:.2f}%",
        "-" * 50,
        f"Win Rate:      {result.win_rate*100:.1f}%",
        f"Avg Odds:      {result.avg_odds:.2f}",
        f"Max Drawdown:  {result.max_drawdown*100:.1f}%",
        f"Sharpe:        {result.sharpe_ratio:.2f}",
        "=" * 50,
    ]
    return "\n".join(lines)
