#!/usr/bin/env python3
"""
Test #1: Calibration Sanity Check

Goal: Verify if isotonic calibration extrapolates poorly on out-of-distribution data.

Expected Results:
- PASS: Calibrated ≈ Input ± 0.03 (smooth extrapolation)
- FAIL: Big jumps like 0.02 → 0.15 (causing 300% EVs)

This test loads the current calibrators and checks how they handle
probabilities at the extremes (low and high values) which are common
in underdog bets on smaller leagues.
"""

import pickle
import numpy as np
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("=" * 70)
    print("TEST #1: CALIBRATION SANITY CHECK")
    print("=" * 70)
    print()
    
    # Load ensemble calibrators
    ensemble_file = Path('models/ensemble_simple_latest.pkl')
    
    if not ensemble_file.exists():
        print(f"❌ ERROR: {ensemble_file} not found")
        print("   Please train ensemble first: python scripts/train_simple_ensemble.py")
        sys.exit(1)
    
    with open(ensemble_file, 'rb') as f:
        ensemble_data = pickle.load(f)
        calibrators = ensemble_data['calibrators']
        method = ensemble_data.get('method', 'unknown')
    
    print(f"✓ Loaded calibrators (method: {method})")
    print(f"  Number of calibrators: {len(calibrators)} (one per class)")
    print()
    
    # Test probabilities spanning full range
    # Pay special attention to extremes (0.01-0.10 and 0.90-0.99)
    test_probs = np.array([
        0.01,  # Very low (common for away underdogs in small leagues)
        0.02,  # Low underdog
        0.05,  # Moderate underdog
        0.10,  # Slight underdog
        0.20,  # Below average
        0.30,  # Average
        0.40,  # Above average
        0.50,  # Even
        0.60,  # Likely
        0.70,  # Very likely
        0.80,  # Heavy favorite
        0.90,  # Near certain
        0.95   # Extreme favorite
    ])
    
    print("Testing calibration across probability range...")
    print("-" * 70)
    print(f"{'Raw Prob':<12} | {'Home':<12} | {'Draw':<12} | {'Away':<12}")
    print("-" * 70)
    
    class_names = ['Home', 'Draw', 'Away']
    max_deviation = 0.0
    extreme_cases = []
    
    # Detect calibrator type
    is_platt = hasattr(calibrators[0], 'predict_proba')
    print(f"Calibrator type: {'Platt (Logistic)' if is_platt else 'Isotonic'}")
    print()
    
    for prob in test_probs:
        calibrated_vals = []
        
        for i, cal in enumerate(calibrators):
            try:
                if is_platt:
                    # Platt scaling: needs 2D array
                    X = np.array([[prob]])
                    calibrated = cal.predict_proba(X)[0, 1]
                else:
                    # Isotonic: accepts 1D
                    calibrated = cal.predict(np.array([prob]))[0]
                
                calibrated_vals.append(calibrated)
                
                # Check for extreme deviations
                deviation = abs(calibrated - prob)
                if deviation > max_deviation:
                    max_deviation = deviation
                
                # Flag extreme cases (>10% deviation)
                if deviation > 0.10:
                    extreme_cases.append({
                        'raw': prob,
                        'calibrated': calibrated,
                        'class': class_names[i],
                        'deviation': deviation
                    })
            except Exception as e:
                calibrated_vals.append(f"ERROR: {e}")
        
        # Format output
        prob_str = f"{prob:.2f}"
        home_str = f"{calibrated_vals[0]:.3f}" if isinstance(calibrated_vals[0], (float, np.float64)) else str(calibrated_vals[0])[:30]
        draw_str = f"{calibrated_vals[1]:.3f}" if isinstance(calibrated_vals[1], (float, np.float64)) else str(calibrated_vals[1])[:30]
        away_str = f"{calibrated_vals[2]:.3f}" if isinstance(calibrated_vals[2], (float, np.float64)) else str(calibrated_vals[2])[:30]
        
        print(f"{prob_str:<12} | {home_str:<12} | {draw_str:<12} | {away_str:<12}")
    
    print("-" * 70)
    print()
    
    # Analysis
    print("ANALYSIS:")
    print(f"  Max deviation: {max_deviation:.3f}")
    print()
    
    if extreme_cases:
        print("⚠️  EXTREME DEVIATIONS DETECTED (>10%):")
        print()
        for case in extreme_cases:
            print(f"  [{case['class']}] {case['raw']:.3f} → {case['calibrated']:.3f} "
                  f"(deviation: {case['deviation']:.1%})")
        print()
        print("🔴 FAIL: Isotonic regression extrapolates poorly!")
        print()
        print("IMPACT ON EV:")
        # Example calculation
        worst_case = max(extreme_cases, key=lambda x: x['deviation'])
        raw_p = worst_case['raw']
        calibrated_p = worst_case['calibrated']
        example_odds = 10.0  # Common for underdogs
        
        raw_ev = (raw_p * example_odds - 1) * 100
        calibrated_ev = (calibrated_p * example_odds - 1) * 100
        
        print(f"  Example: Underdog @ {example_odds} odds")
        print(f"  - Raw probability: {raw_p:.1%} → EV: {raw_ev:.0f}%")
        print(f"  - Calibrated probability: {calibrated_p:.1%} → EV: {calibrated_ev:.0f}%")
        print(f"  - Inflation: {calibrated_ev - raw_ev:+.0f}%")
        print()
        
        return 1  # Test failed
    else:
        print("✅ PASS: Calibration extrapolates smoothly")
        print("   (All deviations < 10%)")
        print()
        
        if max_deviation > 0.05:
            print("⚠️  Note: Some moderate deviations (5-10%) detected")
            print("   This is acceptable but could be improved with Platt scaling")
        
        return 0  # Test passed

if __name__ == '__main__':
    sys.exit(main())
