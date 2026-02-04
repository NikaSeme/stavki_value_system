#!/usr/bin/env python3
"""
Professional Backtest Runner
=============================
Complete backtesting with all validation methods.

Usage:
    python scripts/run_professional_backtest.py [--data PATH] [--config PATH]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
from datetime import datetime

from src.backtesting import BacktestEngine, BacktestConfig, BacktestDataLoader
from src.backtesting.validators import WalkForwardValidator, MonteCarloSimulator
from src.backtesting.reality import RealitySimulator
from src.backtesting.metrics import MetricsCalculator
from src.backtesting.optimizer import BayesianOptimizer
from src.backtesting.reports import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Professional Backtesting")
    parser.add_argument("--data", default="data/processed/multi_league_features_peopled.csv",
                       help="Path to data file")
    parser.add_argument("--leagues", nargs="+", 
                       default=["EPL", "LaLiga", "Bundesliga", "SerieA", "Ligue1"],
                       help="Leagues to include")
    parser.add_argument("--min-ev", type=float, default=0.05,
                       help="Minimum EV threshold")
    parser.add_argument("--kelly", type=float, default=0.20,
                       help="Kelly fraction")
    parser.add_argument("--monte-carlo", type=int, default=10000,
                       help="Number of Monte Carlo simulations")
    parser.add_argument("--walk-forward", action="store_true",
                       help="Run Walk-Forward validation")
    parser.add_argument("--reality", choices=["optimistic", "realistic", "pessimistic", "worst_case"],
                       default="realistic", help="Reality scenario")
    parser.add_argument("--optimize", action="store_true",
                       help="Run Bayesian optimization")
    parser.add_argument("--output", default="reports",
                       help="Output directory for reports")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎯 STAVKI Professional Backtesting System")
    print("="*60)
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    loader = BacktestDataLoader()
    data = loader.load_historical(
        leagues=args.leagues,
    )
    
    if data.empty:
        logger.error("No data loaded!")
        return
        
    logger.info(f"Loaded {len(data)} matches across {args.leagues}")
    
    # Create config
    config = BacktestConfig(
        min_ev=args.min_ev,
        kelly_fraction=args.kelly,
        n_simulations=args.monte_carlo,
        leagues=args.leagues,
    )
    
    # Create engine
    engine = BacktestEngine(config=config)
    
    # === OPTIMIZATION ===
    if args.optimize:
        print("\n" + "-"*60)
        print("🔧 Running Bayesian Optimization...")
        print("-"*60)
        
        optimizer = BayesianOptimizer(
            objective="roi",
            n_iterations=50,
            n_random_starts=15,
        )
        
        opt_result = optimizer.optimize(engine, data)
        
        print(f"\n✅ Best params found:")
        for k, v in opt_result.best_params.items():
            print(f"   {k}: {v}")
        print(f"   Expected ROI: {opt_result.best_score*100:.2f}%")
        
        # Apply best params
        for k, v in opt_result.best_params.items():
            if hasattr(config, k):
                setattr(config, k, v)
    
    # === REALITY SIMULATION ===
    print("\n" + "-"*60)
    print(f"🌍 Running Reality Simulation ({args.reality})...")
    print("-"*60)
    
    reality_sim = RealitySimulator(scenario=args.reality)
    config = reality_sim.adjust_config(config)
    engine = BacktestEngine(config=config)
    
    # === MAIN BACKTEST ===
    print("\n" + "-"*60)
    print("📊 Running Main Backtest...")
    print("-"*60)
    
    result = engine.run(data)
    
    print(f"\n{'='*60}")
    print(f"MAIN BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Total Bets:    {result.total_bets}")
    print(f"Total Stake:   €{result.total_stake:,.2f}")
    print(f"Total Profit:  €{result.total_profit:,.2f}")
    print(f"ROI:           {result.roi*100:.2f}%")
    print(f"Win Rate:      {result.win_rate*100:.1f}%")
    print(f"Max Drawdown:  {result.max_drawdown*100:.1f}%")
    print(f"Sharpe Ratio:  {result.sharpe_ratio:.2f}")
    print(f"{'='*60}\n")
    
    # === MONTE CARLO ===
    print("-"*60)
    print(f"🎲 Running Monte Carlo ({args.monte_carlo:,} simulations)...")
    print("-"*60)
    
    mc_simulator = MonteCarloSimulator(
        n_simulations=args.monte_carlo,
        confidence_level=0.95,
    )
    mc_result = mc_simulator.simulate(result)
    
    print(f"\n95% Confidence Interval for ROI:")
    print(f"   Lower: {mc_result.get('roi_ci_lower', 0)*100:.2f}%")
    print(f"   Upper: {mc_result.get('roi_ci_upper', 0)*100:.2f}%")
    print(f"   P(ROI > 0): {mc_result.get('prob_positive_roi', 0)*100:.1f}%")
    print(f"   VaR (5%): {mc_result.get('var_5', 0)*100:.2f}%")
    
    # Update result with CI
    result.roi_ci_lower = mc_result.get("roi_ci_lower", 0)
    result.roi_ci_upper = mc_result.get("roi_ci_upper", 0)
    
    # === WALK-FORWARD ===
    wf_result = None
    if args.walk_forward:
        print("\n" + "-"*60)
        print("🔄 Running Walk-Forward Validation...")
        print("-"*60)
        
        wf_validator = WalkForwardValidator(
            train_months=6,
            test_months=2,
            step_months=1,
        )
        
        wf_results = wf_validator.validate(data, engine)
        wf_result = wf_validator.get_combined_result(wf_results)
        
        print(f"\nWalk-Forward Results:")
        print(f"   Folds: {wf_result.get('n_folds', 0)}")
        print(f"   Aggregate ROI: {wf_result.get('aggregate_roi', 0)*100:.2f}%")
        print(f"   Consistency: {wf_result.get('consistency', 0)*100:.1f}%")
    
    # === GENERATE REPORT ===
    print("\n" + "-"*60)
    print("📝 Generating Reports...")
    print("-"*60)
    
    generator = ReportGenerator(output_dir=args.output)
    report_paths = generator.generate(
        result=result,
        monte_carlo=mc_result,
        walk_forward=wf_result,
    )
    
    print(f"\n✅ Reports generated:")
    for fmt, path in report_paths.items():
        print(f"   {fmt}: {path}")
    
    # === PER-LEAGUE BREAKDOWN ===
    if result.league_results:
        print("\n" + "-"*60)
        print("🏆 Per-League Results:")
        print("-"*60)
        print(f"{'League':<15} {'Bets':>6} {'Profit':>10} {'ROI':>8} {'WR':>6}")
        print("-"*50)
        for league, stats in sorted(result.league_results.items(), 
                                   key=lambda x: x[1].get("roi", 0), 
                                   reverse=True):
            print(f"{league:<15} {stats.get('bets', 0):>6} €{stats.get('profit', 0):>8.2f} {stats.get('roi', 0)*100:>7.1f}% {stats.get('win_rate', 0)*100:>5.1f}%")
    
    print("\n" + "="*60)
    print("✅ Backtest Complete!")
    print("="*60 + "\n")
    
    return result


if __name__ == "__main__":
    main()
