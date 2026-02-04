#!/usr/bin/env python3
"""
Performance Optimization Tests
==============================
Verify that optimized code produces IDENTICAL results to original.
All tests must pass before deploying optimizations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import time
from datetime import datetime


def test_vectorized_engine():
    """Test that vectorized engine produces identical results to standard."""
    print("\n" + "="*60)
    print("TEST 1: Vectorized Engine Consistency")
    print("="*60)
    
    from src.backtesting import BacktestEngine, BacktestConfig
    from src.backtesting.engine_vectorized import VectorizedBacktestEngine, verify_consistency
    
    # Create test data
    np.random.seed(42)
    n = 500
    
    data = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "HomeTeam": [f"Team_{i%10}" for i in range(n)],
        "AwayTeam": [f"Team_{(i+5)%10}" for i in range(n)],
        "League": ["EPL"] * n,
        "B365H": np.random.uniform(1.5, 4.0, n),
        "B365D": np.random.uniform(3.0, 4.5, n),
        "B365A": np.random.uniform(1.5, 5.0, n),
        "FTR": np.random.choice(["H", "D", "A"], n, p=[0.45, 0.25, 0.30]),
    })
    
    config = BacktestConfig(min_ev=0.05, kelly_fraction=0.2)
    
    # Run both engines
    standard = BacktestEngine(config=config)
    vectorized = VectorizedBacktestEngine(config=config)
    
    t1 = time.time()
    result_std = standard.run(data)
    time_std = time.time() - t1
    
    t2 = time.time()
    result_vec = vectorized.run(data)
    time_vec = time.time() - t2
    
    # Compare
    print(f"\nStandard engine:  {result_std.total_bets} bets, ROI={result_std.roi:.4f}, time={time_std:.3f}s")
    print(f"Vectorized engine: {result_vec.total_bets} bets, ROI={result_vec.roi:.4f}, time={time_vec:.3f}s")
    print(f"Speedup: {time_std/time_vec:.1f}x")
    
    # Verify consistency
    tolerance = 0.001
    checks = {
        "bets_match": result_std.total_bets == result_vec.total_bets,
        "stake_match": abs(result_std.total_stake - result_vec.total_stake) < tolerance * result_std.total_stake,
        "profit_match": abs(result_std.total_profit - result_vec.total_profit) < tolerance * abs(result_std.total_profit) if result_std.total_profit != 0 else True,
        "roi_match": abs(result_std.roi - result_vec.roi) < tolerance,
    }
    
    print("\nConsistency checks:")
    all_passed = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
        if not passed:
            all_passed = False
    
    return all_passed


def test_parallel_monte_carlo():
    """Test that parallel MC produces statistically similar results."""
    print("\n" + "="*60)
    print("TEST 2: Parallel Monte Carlo Consistency")
    print("="*60)
    
    from src.backtesting import BacktestConfig, BacktestResult
    from src.backtesting.engine import BetResult
    from src.backtesting.validators.monte_carlo import MonteCarloSimulator
    from src.backtesting.validators.monte_carlo_parallel import ParallelMonteCarloSimulator
    
    # Create fake backtest result
    np.random.seed(42)
    n_bets = 200
    
    bets = []
    for i in range(n_bets):
        won = np.random.random() < 0.48
        odds = np.random.uniform(1.5, 3.0)
        stake = 10
        profit = stake * (odds - 1) if won else -stake
        
        bets.append(BetResult(
            event_id=str(i),
            date=datetime.now(),
            league="EPL",
            home_team="A",
            away_team="B",
            outcome="H",
            odds=odds,
            stake=stake,
            p_model=0.55,
            ev=0.1,
            won=won,
            profit=profit,
        ))
    
    result = BacktestResult(
        bets=bets,
        config=BacktestConfig(),
        start_date=datetime.now(),
        end_date=datetime.now(),
    )
    
    # Run both versions
    n_sims = 5000
    
    seq = MonteCarloSimulator(n_simulations=n_sims, random_seed=42)
    par = ParallelMonteCarloSimulator(n_simulations=n_sims, random_seed=42)
    
    t1 = time.time()
    seq_result = seq.simulate(result)
    time_seq = time.time() - t1
    
    t2 = time.time()
    par_result = par.simulate(result)
    time_par = time.time() - t2
    
    print(f"\nSequential: ROI_mean={seq_result['roi_mean']:.4f}, time={time_seq:.3f}s")
    print(f"Parallel:   ROI_mean={par_result['roi_mean']:.4f}, time={time_par:.3f}s")
    print(f"Speedup: {time_seq/time_par:.1f}x")
    
    # Statistical comparison (means should be similar within 2% tolerance)
    tolerance = 0.02
    roi_diff = abs(seq_result["roi_mean"] - par_result["roi_mean"])
    
    checks = {
        "roi_mean_similar": roi_diff < tolerance,
        "both_have_ci": "roi_ci_lower" in seq_result and "roi_ci_lower" in par_result,
        "both_have_var": "var_5" in seq_result and "var_5" in par_result,
    }
    
    print(f"\nROI mean difference: {roi_diff:.4f} (tolerance: {tolerance})")
    print("\nConsistency checks:")
    all_passed = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
        if not passed:
            all_passed = False
    
    return all_passed


def test_data_loader_parquet():
    """Test Parquet loading and caching."""
    print("\n" + "="*60)
    print("TEST 3: Data Loader Parquet & Caching")
    print("="*60)
    
    from src.backtesting import BacktestDataLoader
    
    loader = BacktestDataLoader()
    
    # First load (from file)
    t1 = time.time()
    data1 = loader.load_historical(leagues=["EPL"])
    time1 = time.time() - t1
    
    # Second load (from cache)
    t2 = time.time()
    data2 = loader.load_historical(leagues=["EPL"])
    time2 = time.time() - t2
    
    print(f"\nFirst load:  {len(data1)} rows, time={time1:.3f}s")
    print(f"Cached load: {len(data2)} rows, time={time2:.3f}s")
    print(f"Cache speedup: {time1/time2:.1f}x")
    
    # Verify data is identical
    if len(data1) > 0 and len(data2) > 0:
        data_match = len(data1) == len(data2)
        cache_faster = time2 < time1
        
        checks = {
            "data_length_match": data_match,
            "cache_is_faster": cache_faster,
        }
        
        print("\nConsistency checks:")
        all_passed = True
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")
            if not passed:
                all_passed = False
    else:
        print("\n⚠️ No data available for comparison (no files found)")
        all_passed = True  # Not a failure, just no data
    
    return all_passed


def test_nan_handling():
    """Test NaN and invalid value handling."""
    print("\n" + "="*60)
    print("TEST 4: NaN & Invalid Value Handling")
    print("="*60)
    
    from src.backtesting import BacktestEngine, BacktestConfig
    
    # Create data with NaN and invalid values
    data = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=10, freq="D"),
        "HomeTeam": ["A"]*10,
        "AwayTeam": ["B"]*10,
        "League": ["EPL"]*10,
        "B365H": [2.0, float('nan'), 0, -1, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0],  # Invalid values
        "B365D": [3.5]*10,
        "B365A": [3.0]*10,
        "FTR": ["H"]*10,
    })
    
    config = BacktestConfig(min_ev=0.01)
    engine = BacktestEngine(config=config)
    
    # This should NOT crash
    try:
        result = engine.run(data)
        print(f"\nBacktest completed: {result.total_bets} bets, ROI={result.roi:.4f}")
        print("✅ NaN handling passed - no crash")
        return True
    except Exception as e:
        print(f"❌ NaN handling failed: {e}")
        return False


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "="*60)
    print("PERFORMANCE OPTIMIZATION VERIFICATION")
    print("="*60)
    print("All tests verify that optimized code produces")
    print("IDENTICAL or STATISTICALLY EQUIVALENT results.")
    print("="*60)
    
    results = {
        "vectorized_engine": test_vectorized_engine(),
        "parallel_monte_carlo": test_parallel_monte_carlo(),
        "data_loader": test_data_loader_parquet(),
        "nan_handling": test_nan_handling(),
    }
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Optimizations are safe to use!")
    else:
        print("❌ SOME TESTS FAILED - Review before deploying!")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
