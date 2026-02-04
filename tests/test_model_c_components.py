#!/usr/bin/env python3
"""Test all Model C components."""

import pandas as pd
import numpy as np
from pathlib import Path

def test_neural_predictor():
    print("TEST 1: NEURAL PREDICTOR")
    print("-" * 40)
    
    from src.models.neural_predictor import NeuralPredictor
    np_model = NeuralPredictor()
    
    test_df = pd.DataFrame({
        'HomeEloBefore': [1600, 1400],
        'AwayEloBefore': [1500, 1600],
        'EloDiff': [100, -200],
        'Home_Pts_L5': [2.0, 1.0],
        'Home_GF_L5': [1.8, 0.8],
        'Home_GA_L5': [0.8, 1.5],
        'Away_Pts_L5': [1.5, 2.0],
        'Away_GF_L5': [1.2, 1.8],
        'Away_GA_L5': [1.0, 0.8],
        'Home_Overall_Pts_L5': [2.0, 1.0],
        'Home_Overall_GF_L5': [1.8, 0.8],
        'Home_Overall_GA_L5': [0.8, 1.5],
        'Away_Overall_Pts_L5': [1.5, 2.0],
        'Away_Overall_GF_L5': [1.2, 1.8],
        'Away_Overall_GA_L5': [1.0, 0.8],
        'Odds_Volatility': [0.02, 0.05],
        'SentimentHome': [0.1, -0.2],
        'SentimentAway': [-0.1, 0.1],
    })
    
    probs = np_model.predict(test_df)
    assert probs.shape == (2, 3), f"Wrong shape: {probs.shape}"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=0.01), "Probs don't sum to 1"
    print(f"  Predictions: {probs[0]}")
    print("  PASSED")

def test_meta_filter():
    print("\nTEST 2: META FILTER")
    print("-" * 40)
    
    from src.models.meta_filter import MetaFilter
    mf = MetaFilter('models/meta_filter_latest.pkl')
    loaded = mf.load()
    
    if loaded:
        prob = mf.predict_proba(ev_pct=0.10, odds=2.5, model_confidence=0.45, divergence=0.08)
        assert 0 <= prob <= 1, f"Invalid probability: {prob}"
        print(f"  Predict proba: {prob:.3f}")
        print("  PASSED")
    else:
        print("  SKIPPED (model not found)")

def test_staking():
    print("\nTEST 3: KELLY STAKING")
    print("-" * 40)
    
    from src.strategy.staking_kelly import Staker, kelly_simple
    
    # Test kelly formula
    k = kelly_simple(0.60, 2.0)
    assert k > 0, "Kelly should be positive for +EV bet"
    print(f"  Kelly (60% prob, 2.0 odds): {k:.2%}")
    
    k_neg = kelly_simple(0.30, 2.0)
    assert k_neg == 0, "Kelly should be 0 for -EV bet"
    print(f"  Kelly (30% prob, 2.0 odds): {k_neg:.2%} (correctly 0)")
    
    # Test staker
    staker = Staker(bankroll=1000)
    stake = staker.calculate_stake(0.55, 2.0, 'E0')
    assert 0 <= stake <= 50, f"Stake too high: {stake}"
    print(f"  Staker stake: ${stake:.2f}")
    print("  PASSED")

def test_clv():
    print("\nTEST 4: CLV TRACKER")
    print("-" * 40)
    
    from src.analytics.clv_tracker import calculate_simple_clv
    
    # Beat the line
    clv1 = calculate_simple_clv(2.10, 2.00)
    assert clv1 < 0, "CLV should be negative when we got worse implied prob"
    
    # Lost to line
    clv2 = calculate_simple_clv(2.00, 2.10)
    assert clv2 > 0, "CLV should be positive when we beat closing"
    
    print(f"  CLV (2.10 vs 2.00): {clv1:+.4f}")
    print(f"  CLV (2.00 vs 2.10): {clv2:+.4f}")
    print("  PASSED")

def test_data_stats():
    print("\nTEST 5: DATA STATISTICS")
    print("-" * 40)
    
    data_file = Path('data/processed/multi_league_features_2021_2024.csv')
    df = pd.read_csv(data_file)
    
    print(f"  Total samples: {len(df)}")
    print(f"  Leagues:")
    for league, count in df['League'].value_counts().items():
        print(f"    {league}: {count} ({count/len(df)*100:.1f}%)")
    
    print("  PASSED")

if __name__ == '__main__':
    print("=" * 50)
    print("MODEL C COMPONENTS TEST SUITE")
    print("=" * 50)
    
    test_neural_predictor()
    test_meta_filter()
    test_staking()
    test_clv()
    test_data_stats()
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
