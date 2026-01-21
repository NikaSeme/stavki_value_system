"""
Unit tests for feature engineering module.

CRITICAL: Includes data leakage prevention tests.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    build_features_dataset,
    build_match_features,
    calculate_points,
    calculate_team_form,
)


class TestPointsCalculation:
    """Test points calculation logic."""
    
    def test_win(self):
        """Test win gives 3 points."""
        assert calculate_points(2, 1) == 3
        assert calculate_points(3, 0) == 3
    
    def test_draw(self):
        """Test draw gives 1 point."""
        assert calculate_points(1, 1) == 1
        assert calculate_points(0, 0) == 1
    
    def test_loss(self):
        """Test loss gives 0 points."""
        assert calculate_points(0, 1) == 0
        assert calculate_points(1, 3) == 0


class TestTeamFormCalculation:
    """Test team form calculation."""
    
    def test_basic_form_calculation(self):
        """Test basic rolling statistics."""
        # Create test matches
        matches = pd.DataFrame({
            'date': pd.to_datetime([
                '2025-08-01',
                '2025-08-02',
                '2025-08-03',
            ]),
            'league': ['E0', 'E0', 'E0'],
            'home_team': ['Arsenal', 'Chelsea', 'Arsenal'],
            'away_team': ['Chelsea', 'Liverpool', 'Man City'],
            'home_goals': [2, 1, 1],
            'away_goals': [1, 1, 2],
        })
        
        # Calculate Arsenal's form before 2025-08-04
        # Should use matches from 08-01 and 08-03 (Arsenal involved)
        form = calculate_team_form(
            matches,
            team='Arsenal',
            before_date=datetime(2025, 8, 4),
            window=5
        )
        
        # Arsenal's history:
        # 08-01: Home vs Chelsea, 2-1 (W, 3pts)
        # 08-03: Home vs Man City, 1-2 (L, 0pts)
        assert form['matches_count'] == 2
        assert form['goals_for_avg'] == 1.5  # (2 + 1) / 2
        assert form['goals_against_avg'] == 1.5  # (1 + 2) / 2
        assert form['points_avg'] == 1.5  # (3 + 0) / 2
        assert form['form_points'] == 3.0  # 3 + 0
    
    def test_away_matches_included(self):
        """Test that away matches are included in form calculation."""
        matches = pd.DataFrame({
            'date': pd.to_datetime([
                '2025-08-01',
                '2025-08-02',
            ]),
            'league': ['E0', 'E0'],
            'home_team': ['Chelsea', 'Liverpool'],
            'away_team': ['Arsenal', 'Arsenal'],
            'home_goals': [1, 0],
            'away_goals': [2, 3],
        })
        
        # Arsenal played 2 away matches
        form = calculate_team_form(
            matches,
            team='Arsenal',
            before_date=datetime(2025, 8, 3),
            window=5
        )
        
        assert form['matches_count'] == 2
        assert form['goals_for_avg'] == 2.5  # (2 + 3) / 2
        assert form['goals_against_avg'] == 0.5  # (1 + 0) / 2
    
    def test_no_history(self):
        """Test team with no history returns NaN."""
        matches = pd.DataFrame({
            'date': pd.to_datetime(['2025-08-01']),
            'league': ['E0'],
            'home_team': ['Chelsea'],
            'away_team': ['Liverpool'],
            'home_goals': [1],
            'away_goals': [1],
        })
        
        # Arsenal has no matches
        form = calculate_team_form(
            matches,
            team='Arsenal',
            before_date=datetime(2025, 8, 2),
            window=5
        )
        
        assert form['matches_count'] == 0
        assert np.isnan(form['goals_for_avg'])
        assert np.isnan(form['points_avg'])
    
    def test_window_limit(self):
        """Test that window size is respected."""
        # Create 10 matches for Arsenal
        dates = [datetime(2025, 8, i) for i in range(1, 11)]
        matches = pd.DataFrame({
            'date': dates,
            'league': ['E0'] * 10,
            'home_team': ['Arsenal'] * 10,
            'away_team': [f'Team{i}' for i in range(10)],
            'home_goals': [2] * 10,
            'away_goals': [1] * 10,
        })
        
        # Request 5-match window
        form = calculate_team_form(
            matches,
            team='Arsenal',
            before_date=datetime(2025, 8, 11),
            window=5
        )
        
        # Should use only last 5 matches (08-06 to 08-10)
        assert form['matches_count'] == 5


class TestDataLeakagePrevention:
    """
    CRITICAL TESTS: Ensure no future information leaks into features.
    """
    
    def test_no_current_match_in_features(self):
        """
        Verify that current match is NOT included in its own features.
        This is the most critical test for data leakage prevention.
        """
        # Create matches with known dates
        matches = pd.DataFrame({
            'date': pd.to_datetime([
                '2025-08-01',
                '2025-08-02',
                '2025-08-03',  # Target match
                '2025-08-04',
            ]),
            'league': ['E0'] * 4,
            'home_team': ['Arsenal'] * 4,
            'away_team': ['Chelsea', 'Liverpool', 'Man City', 'Tottenham'],
            'home_goals': [2, 1, 3, 0],  # Match on 08-03 has 3 goals
            'away_goals': [1, 1, 0, 2],
        })
        
        # Calculate form for match on 2025-08-03
        # Should use ONLY 08-01 and 08-02 (NOT 08-03 itself)
        form = calculate_team_form(
            matches,
            team='Arsenal',
            before_date=datetime(2025, 8, 3),
            window=5
        )
        
        # Verify only 2 matches used (08-01, 08-02)
        assert form['matches_count'] == 2
        
        # Goals average should be (2 + 1) / 2 = 1.5
        # If 08-03 was included, it would be (2 + 1 + 3) / 3 = 2.0
        assert form['goals_for_avg'] == 1.5
        
        # This confirms match on 08-03 is NOT in its own features
    
    def test_no_future_matches_leak(self):
        """
        Verify that matches AFTER target date are never used.
        """
        matches = pd.DataFrame({
            'date': pd.to_datetime([
                '2025-08-01',
                '2025-08-02',
                '2025-08-03',  # Target date
                '2025-08-04',  # Future - should not be used
                '2025-08-05',  # Future - should not be used
            ]),
            'league': ['E0'] * 5,
            'home_team': ['Arsenal'] * 5,
            'away_team': ['Team' + str(i) for i in range(5)],
            'home_goals': [1, 1, 999, 5, 5],  # Future has abnormal values
            'away_goals': [0, 0, 0, 0, 0],
        })
        
        # Calculate form for 08-03
        form = calculate_team_form(
            matches,
            team='Arsenal',
            before_date=datetime(2025, 8, 3),
            window=5
        )
        
        # Should only use 08-01 and 08-02
        assert form['matches_count'] == 2
        assert form['goals_for_avg'] == 1.0
        
        # If future matches leaked, average would be much higher
        assert form['goals_for_avg'] < 2.0
    
    def test_strict_temporal_ordering(self):
        """
        Test that date comparison is strictly < (not <=).
        """
        matches = pd.DataFrame({
            'date': pd.to_datetime([
                '2025-08-01',
                '2025-08-02',
            ]),
            'league': ['E0'] * 2,
            'home_team': ['Arsenal'] * 2,
            'away_team': ['Team1', 'Team2'],
            'home_goals': [2, 3],
            'away_goals': [1, 1],
        })
        
        # Calculate form with before_date = 2025-08-02
        # Should use ONLY 08-01, NOT 08-02 (strict <)
        form = calculate_team_form(
            matches,
            team='Arsenal',
            before_date=datetime(2025, 8, 2),
            window=5
        )
        
        assert form['matches_count'] == 1
        assert form['goals_for_avg'] == 2.0  # Only 08-01


class TestMatchFeaturesBuilding:
    """Test full match features building."""
    
    def test_build_features_preserves_matches(self):
        """Test that all matches are preserved in output."""
        matches = pd.DataFrame({
            'date': ['2025-08-01', '2025-08-02', '2025-08-03'],
            'league': ['E0'] * 3,
            'home_team': ['Arsenal', 'Chelsea', 'Liverpool'],
            'away_team': ['Chelsea', 'Liverpool', 'Man City'],
            'home_goals': [2, 1, 0],
            'away_goals': [1, 1, 2],
            'odds_1': [1.8, 2.0, 2.5],
            'odds_x': [3.5, 3.4, 3.2],
            'odds_2': [4.0, 3.8, 2.8],
        })
        
        features_df = build_match_features(matches, window=5)
        
        # All matches should be in output
        assert len(features_df) == 3
        
        # Original columns should be preserved
        assert 'date' in features_df.columns
        assert 'home_team' in features_df.columns
        assert 'odds_1' in features_df.columns
        
        # Feature columns should be added
        assert 'home_goals_for_avg_5' in features_df.columns
        assert 'away_points_avg_5' in features_df.columns
    
    def test_first_match_has_nan_features(self):
        """Test that first match has NaN features (no history)."""
        matches = pd.DataFrame({
            'date': ['2025-08-01', '2025-08-02'],
            'league': ['E0'] * 2,
            'home_team': ['Arsenal', 'Arsenal'],
            'away_team': ['Chelsea', 'Liverpool'],
            'home_goals': [2, 1],
            'away_goals': [1, 1],
        })
        
        features_df = build_match_features(matches, window=5)
        
        # First match should have NaN features (no history)
        first_match = features_df.iloc[0]
        assert np.isnan(first_match['home_goals_for_avg_5'])
        assert first_match['home_matches_count'] == 0
        
        # Second match should have features from first match
        second_match = features_df.iloc[1]
        assert second_match['home_matches_count'] == 1
        assert second_match['home_goals_for_avg_5'] == 2.0


class TestEndToEnd:
    """Test complete pipeline."""
    
    def test_full_pipeline(self, tmp_path):
        """Test full feature building pipeline."""
        # Create input CSV
        input_file = tmp_path / "matches.csv"
        matches = pd.DataFrame({
            'date': ['2025-08-01', '2025-08-02', '2025-08-03'],
            'league': ['E0'] * 3,
            'home_team': ['Arsenal', 'Chelsea', 'Arsenal'],
            'away_team': ['Chelsea', 'Liverpool', 'Man City'],
            'home_goals': [2, 1, 0],
            'away_goals': [1, 1, 2],
            'odds_1': [1.8, 2.0, 2.5],
            'odds_x': [3.5, 3.4, 3.2],
            'odds_2': [4.0, 3.8, 2.8],
        })
        matches.to_csv(input_file, index=False)
        
        output_file = tmp_path / "features.csv"
        
        # Run pipeline
        stats = build_features_dataset(input_file, output_file, window=5)
        
        # Verify output file exists
        assert output_file.exists()
        
        # Load and verify
        features_df = pd.read_csv(output_file)
        assert len(features_df) == 3
        assert 'home_goals_for_avg_5' in features_df.columns
        
        # Verify stats
        assert stats['total_matches'] == 3
