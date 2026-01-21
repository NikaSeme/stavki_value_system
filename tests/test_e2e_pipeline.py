"""
End-to-end pipeline tests.

Tests the complete pipeline from features + odds to recommendations.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.run_pipeline import run_pipeline, save_recommendations


class TestPipeline:
    """Test pipeline functionality."""
    
    @pytest.fixture
    def features_df(self):
        """Sample features DataFrame."""
        return pd.DataFrame({
            'match_id': [1, 2, 3],
            'date': ['2025-01-15', '2025-01-16', '2025-01-17'],
            'home_team': ['Arsenal', 'Man City', 'Tottenham'],
            'away_team': ['Chelsea', 'Liverpool', 'Man Utd'],
            'home_goals_for_avg_5': [2.0, 2.5, 1.8],
            'home_goals_against_avg_5': [1.0, 0.8, 1.5],
            'home_points_avg_5': [2.5, 2.8, 1.8],
            'home_form_points_5': [12, 14, 9],
            'home_matches_count': [5, 5, 5],
            'away_goals_for_avg_5': [1.5, 2.0, 1.6],
            'away_goals_against_avg_5': [1.2, 1.0, 1.4],
            'away_points_avg_5': [2.0, 2.4, 1.6],
            'away_form_points_5': [10, 12, 8],
            'away_matches_count': [5, 5, 5],
        })
    
    @pytest.fixture
    def odds_df(self):
        """Sample odds DataFrame."""
        return pd.DataFrame({
            'match_id': [1, 2, 3],
            'odds_home': [2.5, 1.8, 2.2],
            'odds_draw': [3.2, 3.5, 3.0],
            'odds_away': [2.8, 4.5, 3.4],
        })
    
    def test_pipeline_runs_successfully(self, features_df, odds_df):
        """Test that pipeline runs without errors."""
        recommendations = run_pipeline(
            features_df,
            odds_df,
            bankroll=1000.0,
            kelly_fraction=0.5
        )
        
        # Should return a DataFrame
        assert isinstance(recommendations, pd.DataFrame)
    
    def test_pipeline_output_columns(self, features_df, odds_df):
        """Test that output has required columns."""
        recommendations = run_pipeline(
            features_df,
            odds_df,
            bankroll=1000.0
        )
        
        required_cols = [
            'match_id', 'outcome', 'probability', 'odds', 'ev', 'stake'
        ]
        
        for col in required_cols:
            assert col in recommendations.columns, f"Missing column: {col}"
    
    def test_stakes_non_negative(self, features_df, odds_df):
        """Test that all stakes are non-negative."""
        recommendations = run_pipeline(
            features_df,
            odds_df,
            bankroll=1000.0
        )
        
        if len(recommendations) > 0:
            assert (recommendations['stake'] >= 0).all()
    
    def test_stakes_within_bankroll(self, features_df, odds_df):
        """Test that total stakes don't exceed bankroll."""
        bankroll = 1000.0
        recommendations = run_pipeline(
            features_df,
            odds_df,
            bankroll=bankroll
        )
        
        if len(recommendations) > 0:
            total_stake = recommendations['stake'].sum()
            # Allow small margin for rounding
            assert total_stake <= bankroll * 1.01
    
    def test_probabilities_valid_range(self, features_df, odds_df):
        """Test that probabilities are in [0, 1]."""
        recommendations = run_pipeline(
            features_df,
            odds_df,
            bankroll=1000.0
        )
        
        if len(recommendations) > 0:
            assert (recommendations['probability'] >= 0).all()
            assert (recommendations['probability'] <= 1).all()
    
    def test_save_recommendations(self, features_df, odds_df):
        """Test saving recommendations to files."""
        recommendations = run_pipeline(
            features_df,
            odds_df,
            bankroll=1000.0
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            paths = save_recommendations(recommendations, output_dir)
            
            # Check files exist
            assert paths['csv'].exists()
            assert paths['json'].exists()
            
            # Check CSV can be read back
            loaded = pd.read_csv(paths['csv'])
            assert len(loaded) == len(recommendations)


class TestPipelineInputs:
    """Test pipeline with various inputs."""
    
    def test_pipeline_with_fixtures(self):
        """Test pipeline with fixture files."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        
        features_df = pd.read_csv(fixtures_dir / "features.csv")
        odds_df = pd.read_csv(fixtures_dir / "odds.csv")
        
        recommendations = run_pipeline(
            features_df,
            odds_df,
            bankroll=1000.0
        )
        
        assert isinstance(recommendations, pd.DataFrame)
    
    def test_pipeline_with_small_bankroll(self):
        """Test pipeline with small bankroll."""
        features_df = pd.DataFrame({
            'match_id': [1],
            'home_goals_for_avg_5': [2.0],
            'home_goals_against_avg_5': [1.0],
            'home_points_avg_5': [2.5],
            'home_form_points_5': [12],
            'home_matches_count': [5],
            'away_goals_for_avg_5': [1.5],
            'away_goals_against_avg_5': [1.2],
            'away_points_avg_5': [2.0],
            'away_form_points_5': [10],
            'away_matches_count': [5],
        })
        
        odds_df = pd.DataFrame({
            'match_id': [1],
            'odds_home': [2.5],
            'odds_draw': [3.2],
            'odds_away': [2.8],
        })
        
        recommendations = run_pipeline(
            features_df,
            odds_df,
            bankroll=100.0,  # Small bankroll
            max_stake_fraction=0.1
        )
        
        if len(recommendations) > 0:
            # Max stake should be 10% of 100 = 10
            assert recommendations['stake'].max() <= 10.0


class TestCLIIntegration:
    """Test CLI command integration."""
    
    def test_cli_run_command_exists(self):
        """Test that run command is registered."""
        from src.cli import cli
        
        # Check command exists
        assert 'run' in cli.commands
    
    def test_cli_run_with_fixtures(self):
        """Test run command with fixture files."""
        from click.testing import CliRunner
        from src.cli import cli
        
        runner = CliRunner()
        
        fixtures_dir = Path(__file__).parent / "fixtures"
        features_path = fixtures_dir / "features.csv"
        odds_path = fixtures_dir / "odds.csv"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                'run',
                '--features', str(features_path),
                '--odds', str(odds_path),
                '--bankroll', '1000',
                '--output', tmpdir
            ])
            
            # Command should succeed
            assert result.exit_code == 0
            
            # Check new production files were created (T080)
            output_dir = Path(tmpdir)
            assert (output_dir / 'bets.json').exists()
            assert (output_dir / 'bets.txt').exists()
