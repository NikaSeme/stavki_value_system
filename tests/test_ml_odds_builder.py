#!/usr/bin/env python3
"""
Unit tests for ML pipeline fixes (Task A).

Tests:
1. Draw recognition (Draw, Tie, X)
2. Outcome classification
3. Event skipping for missing outcomes
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.ml_odds_builder import (
    is_draw_outcome,
    classify_outcome,
    extract_hda_odds_from_bookmaker,
    build_ml_odds_line,
    DRAW_NAMES
)


class TestDrawRecognition:
    """Test draw outcome recognition."""
    
    def test_draw_lowercase(self):
        """'draw' → recognized as draw."""
        assert is_draw_outcome("draw") is True
    
    def test_draw_capitalized(self):
        """'Draw' → recognized as draw."""
        assert is_draw_outcome("Draw") is True
    
    def test_draw_uppercase(self):
        """'DRAW' → recognized as draw."""
        assert is_draw_outcome("DRAW") is True
    
    def test_tie_lowercase(self):
        """'tie' → recognized as draw."""
        assert is_draw_outcome("tie") is True
    
    def test_tie_capitalized(self):
        """'Tie' → recognized as draw."""
        assert is_draw_outcome("Tie") is True
    
    def test_x_lowercase(self):
        """'x' → recognized as draw."""
        assert is_draw_outcome("x") is True
    
    def test_x_uppercase(self):
        """'X' → recognized as draw."""
        assert is_draw_outcome("X") is True
    
    def test_draw_with_whitespace(self):
        """' Draw ' with whitespace → recognized as draw."""
        assert is_draw_outcome("  Draw  ") is True
    
    def test_home_team_not_draw(self):
        """Team name → NOT recognized as draw."""
        assert is_draw_outcome("Manchester United") is False
    
    def test_away_not_draw(self):
        """'Away' → NOT recognized as draw."""
        assert is_draw_outcome("Away") is False


class TestOutcomeClassification:
    """Test outcome classification."""
    
    def test_home_team_exact_match(self):
        """Home team exact match → 'H'."""
        result = classify_outcome("Arsenal", "Arsenal", "Chelsea")
        assert result == 'H'
    
    def test_away_team_exact_match(self):
        """Away team exact match → 'A'."""
        result = classify_outcome("Chelsea", "Arsenal", "Chelsea")
        assert result == 'A'
    
    def test_draw_exact(self):
        """'Draw' → 'D'."""
        result = classify_outcome("Draw", "Arsenal", "Chelsea")
        assert result == 'D'
    
    def test_tie_lowercase(self):
        """'tie' → 'D'."""
        result = classify_outcome("tie", "Arsenal", "Chelsea")
        assert result == 'D'
    
    def test_x_draw(self):
        """'X' → 'D'."""
        result = classify_outcome("X", "Arsenal", "Chelsea")
        assert result == 'D'
    
    def test_unknown_outcome(self):
        """Unknown outcome → None."""
        result = classify_outcome("Some Random Text", "Arsenal", "Chelsea")
        assert result is None
    
    def test_case_insensitive_teams(self):
        """Team matching is case-insensitive."""
        result = classify_outcome("arsenal", "Arsenal", "Chelsea")
        assert result == 'H'


class TestExtractHDAOdds:
    """Test H/D/A odds extraction from bookmaker data."""
    
    def test_complete_line(self):
        """Complete H/D/A line → tuple of 3 odds."""
        bk_data = {
            "key": "test",
            "markets": [{
                "key": "h2h",
                "outcomes": [
                    {"name": "Arsenal", "price": 2.0},
                    {"name": "Draw", "price": 3.5},
                    {"name": "Chelsea", "price": 3.0},
                ]
            }]
        }
        
        result = extract_hda_odds_from_bookmaker(bk_data, "Arsenal", "Chelsea")
        assert result is not None
        assert result == (2.0, 3.5, 3.0)  # H, D, A
    
    def test_missing_draw(self):
        """Missing draw outcome → None (event skipped)."""
        bk_data = {
            "key": "test",
            "markets": [{
                "key": "h2h",
                "outcomes": [
                    {"name": "Arsenal", "price": 2.0},
                    {"name": "Chelsea", "price": 3.0},
                    # No draw!
                ]
            }]
        }
        
        result = extract_hda_odds_from_bookmaker(bk_data, "Arsenal", "Chelsea")
        assert result is None
    
    def test_missing_home(self):
        """Missing home outcome → None (event skipped)."""
        bk_data = {
            "key": "test",
            "markets": [{
                "key": "h2h",
                "outcomes": [
                    {"name": "Draw", "price": 3.5},
                    {"name": "Chelsea", "price": 3.0},
                    # No home!
                ]
            }]
        }
        
        result = extract_hda_odds_from_bookmaker(bk_data, "Arsenal", "Chelsea")
        assert result is None
    
    def test_missing_away(self):
        """Missing away outcome → None (event skipped)."""
        bk_data = {
            "key": "test",
            "markets": [{
                "key": "h2h",
                "outcomes": [
                    {"name": "Arsenal", "price": 2.0},
                    {"name": "Draw", "price": 3.5},
                    # No away!
                ]
            }]
        }
        
        result = extract_hda_odds_from_bookmaker(bk_data, "Arsenal", "Chelsea")
        assert result is None
    
    def test_no_h2h_market(self):
        """No h2h market → None."""
        bk_data = {
            "key": "test",
            "markets": [{
                "key": "spreads",  # Wrong market
                "outcomes": [
                    {"name": "Arsenal", "price": 2.0},
                ]
            }]
        }
        
        result = extract_hda_odds_from_bookmaker(bk_data, "Arsenal", "Chelsea")
        assert result is None
    
    def test_tie_as_draw(self):
        """'Tie' outcome → recognized as draw."""
        bk_data = {
            "key": "test",
            "markets": [{
                "key": "h2h",
                "outcomes": [
                    {"name": "Arsenal", "price": 2.0},
                    {"name": "Tie", "price": 3.5},  # Tie instead of Draw
                    {"name": "Chelsea", "price": 3.0},
                ]
            }]
        }
        
        result = extract_hda_odds_from_bookmaker(bk_data, "Arsenal", "Chelsea")
        assert result is not None
        assert result[1] == 3.5


class TestBuildMLOddsLine:
    """Test ML odds line building."""
    
    def test_pinnacle_first(self):
        """Pinnacle available → use Pinnacle line."""
        event = {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "bookmakers": [
                {
                    "key": "bet365",
                    "markets": [{
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Arsenal", "price": 2.1},
                            {"name": "Draw", "price": 3.6},
                            {"name": "Chelsea", "price": 3.1},
                        ]
                    }]
                },
                {
                    "key": "pinnacle",
                    "markets": [{
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Arsenal", "price": 2.0},
                            {"name": "Draw", "price": 3.5},
                            {"name": "Chelsea", "price": 3.0},
                        ]
                    }]
                },
            ]
        }
        
        result = build_ml_odds_line(event)
        assert result is not None
        assert result.source == "pinnacle"
        assert result.home_odds == 2.0
        assert result.draw_odds == 3.5
        assert result.away_odds == 3.0
    
    def test_median_fallback(self):
        """No Pinnacle → use median consensus."""
        event = {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "bookmakers": [
                {
                    "key": "bet365",
                    "markets": [{
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Arsenal", "price": 2.0},
                            {"name": "Draw", "price": 3.5},
                            {"name": "Chelsea", "price": 3.0},
                        ]
                    }]
                },
                {
                    "key": "unibet",
                    "markets": [{
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Arsenal", "price": 2.2},
                            {"name": "Draw", "price": 3.7},
                            {"name": "Chelsea", "price": 3.2},
                        ]
                    }]
                },
            ]
        }
        
        result = build_ml_odds_line(event)
        assert result is not None
        assert result.source == "median"
        # Median of [2.0, 2.2] = 2.1
        assert result.home_odds == 2.1
    
    def test_no_valid_lines(self):
        """No complete H/D/A lines → returns None (event skipped)."""
        event = {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "bookmakers": [
                {
                    "key": "bet365",
                    "markets": [{
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Arsenal", "price": 2.0},
                            # Missing Draw and Away!
                        ]
                    }]
                },
            ]
        }
        
        result = build_ml_odds_line(event)
        assert result is None
    
    def test_overround_calculation(self):
        """Overround calculated correctly."""
        event = {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "bookmakers": [{
                "key": "pinnacle",
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Arsenal", "price": 2.0},   # 50%
                        {"name": "Draw", "price": 4.0},       # 25%
                        {"name": "Chelsea", "price": 4.0},    # 25%
                    ]
                }]
            }]
        }
        
        result = build_ml_odds_line(event)
        assert result is not None
        # 0.5 + 0.25 + 0.25 = 1.0 (no vig in this case)
        assert abs(result.overround - 1.0) < 0.001


def run_tests():
    """Run all tests with pytest."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
