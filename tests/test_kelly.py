"""
Unit tests for Kelly criterion and EV calculations.

Tests:
- EV formula correctness
- Kelly stake non-negative and capped
- Edge cases: odds <=1, invalid probabilities
"""

import numpy as np
import pytest

from src.strategy.ev import calculate_ev, filter_positive_ev
from src.strategy.staking import fractional_kelly, kelly_stake


class TestEVCalculation:
    """Test Expected Value calculations."""
    
    def test_ev_formula_positive(self):
        """Test EV = p*odds - 1 for positive EV."""
        # 50% chance, 2.5 odds → EV = 0.5 * 2.5 - 1 = 0.25
        ev = calculate_ev(0.5, 2.5)
        assert ev == pytest.approx(0.25)
    
    def test_ev_formula_negative(self):
        """Test EV for negative EV bet."""
        # 40% chance, 2.0 odds → EV = 0.4 * 2.0 - 1 = -0.2
        ev = calculate_ev(0.4, 2.0)
        assert ev == pytest.approx(-0.2)
    
    def test_ev_fair_odds(self):
        """Test EV = 0 for fair odds."""
        # 50% chance, 2.0 odds → EV = 0.5 * 2.0 - 1 = 0.0
        ev = calculate_ev(0.5, 2.0)
        assert ev == pytest.approx(0.0)
    
    def test_ev_edge_case_invalid_probability(self):
        """Test invalid probability returns NaN."""
        ev_high = calculate_ev(1.5, 2.0)  # > 1
        ev_low = calculate_ev(-0.1, 2.0)  # < 0
        
        assert np.isnan(ev_high)
        assert np.isnan(ev_low)
    
    def test_ev_edge_case_invalid_odds(self):
        """Test odds <= 1 returns NaN."""
        ev_one = calculate_ev(0.5, 1.0)
        ev_low = calculate_ev(0.5, 0.5)
        
        assert np.isnan(ev_one)
        assert np.isnan(ev_low)
    
    def test_ev_vectorized(self):
        """Test EV calculation on arrays."""
        probs = np.array([0.5, 0.6, 0.4])
        odds = np.array([2.0, 2.5, 3.0])
        
        evs = calculate_ev(probs, odds)
        
        expected = np.array([0.0, 0.5, 0.2])
        np.testing.assert_array_almost_equal(evs, expected)


class TestKellyStaking:
    """Test Kelly Criterion staking."""
    
    def test_kelly_positive_ev(self):
        """Test Kelly stake for positive EV bet."""
        # 55% probability, 2.0 odds, 1000 bankroll, full Kelly
        stake = kelly_stake(0.55, 2.0, 1000.0, kelly_fraction=1.0, max_stake_fraction=1.0)
        
        # Kelly: f = (0.55 * 2.0 - 1) / (2.0 - 1) = 0.1
        # Stake = 0.1 * 1000 = 100
        assert stake == pytest.approx(100.0)
    
    def test_kelly_negative_ev(self):
        """Test Kelly stake = 0 for negative EV."""
        # 40% probability, 2.0 odds → negative EV
        stake = kelly_stake(0.40, 2.0, 1000.0)
        
        assert stake == 0.0
    
    def test_fractional_kelly(self):
        """Test fractional Kelly (half Kelly)."""
        # 55% probability, 2.0 odds, half Kelly
        stake = kelly_stake(0.55, 2.0, 1000.0, kelly_fraction=0.5)
        
        # Full Kelly = 100, half Kelly = 50
        assert stake == pytest.approx(50.0)
    
    def test_kelly_capped_at_max(self):
        """Test Kelly stake capped at maximum."""
        # Very high edge, should hit cap
        stake = kelly_stake(0.8, 3.0, 1000.0, kelly_fraction=1.0, max_stake_fraction=0.05)
        
        # Cap = 5% of 1000 = 50
        assert stake == pytest.approx(50.0)
    
    def test_kelly_non_negative(self):
        """Test Kelly stake is never negative."""
        # Various scenarios
        stakes = [
            kelly_stake(0.3, 2.0, 1000.0),  # Negative EV
            kelly_stake(0.0, 2.0, 1000.0),  # Zero probability
            kelly_stake(0.5, 1.5, 1000.0),  # Low odds
        ]
        
        for stake in stakes:
            assert stake >= 0.0
    
    def test_kelly_edge_case_odds_equal_one(self):
        """Test odds = 1.0 returns 0 stake."""
        stake = kelly_stake(0.5, 1.0, 1000.0)
        assert stake == 0.0
    
    def test_kelly_edge_case_prob_zero(self):
        """Test probability = 0 returns 0 stake."""
        stake = kelly_stake(0.0, 3.0, 1000.0)
        assert stake == 0.0
    
    def test_kelly_edge_case_prob_one(self):
        """Test probability = 1 returns 0 stake (invalid)."""
        stake = kelly_stake(1.0, 2.0, 1000.0)
        assert stake == 0.0
    
    def test_kelly_vectorized(self):
        """Test Kelly on arrays."""
        probs = np.array([0.55, 0.60, 0.40])
        odds = np.array([2.0, 2.5, 2.0])
        
        stakes = kelly_stake(probs, odds, 1000.0, kelly_fraction=0.5)
        
        # All should be non-negative
        assert (stakes >= 0).all()
        
        # Last one (negative EV) should be 0
        assert stakes[2] == 0.0


class TestFractionalKelly:
    """Test fractional Kelly wrapper."""
    
    def test_fractional_kelly_wrapper(self):
        """Test fractional_kelly is equivalent to kelly_stake."""
        prob = 0.55
        odds = 2.0
        
        stake1 = fractional_kelly(prob, odds, 1000.0, fraction=0.5, max_stake_pct=5.0)
        stake2 = kelly_stake(prob, odds, 1000.0, kelly_fraction=0.5, max_stake_fraction=0.05)
        
        assert stake1 == pytest.approx(stake2)
    
    def test_fractional_kelly_percentage_max(self):
        """Test max_stake_pct as percentage."""
        # High edge, should hit 5% cap
        stake = fractional_kelly(0.8, 3.0, 1000.0, fraction=1.0, max_stake_pct=5.0)
        
        assert stake == pytest.approx(50.0)  # 5% of 1000


class TestEdgeCases:
    """Test comprehensive edge cases."""
    
    def test_all_invalid_inputs(self):
        """Test various invalid inputs."""
        invalid_cases = [
            (1.5, 2.0),   # probability > 1
            (-0.1, 2.0),  # probability < 0
            (0.5, 0.5),   # odds < 1
            (0.5, 1.0),   # odds = 1
            (np.nan, 2.0),  # NaN probability
            (0.5, np.nan),  # NaN odds
        ]
        
        for prob, odds in invalid_cases:
            stake = kelly_stake(prob, odds, 1000.0)
            assert stake == 0.0, f"Failed for prob={prob}, odds={odds}"
