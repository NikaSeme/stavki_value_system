"""
Tests for production CLI and reporting (T080).

Tests:
- CLI creates output files
- Stake validations
- EV threshold filtering
- Warnings generation
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from src.cli import cli
from src.pipeline.reports import collect_warnings, generate_report


class TestProductionReporting:
    """Test production reporting functionality."""
    
    @pytest.fixture
    def sample_recommendations(self):
        """Sample recommendations DataFrame."""
        return pd.DataFrame({
            'match_id': [1, 2, 3],
            'date': ['2025-01-20', '2025-01-20', '2025-01-21'],
            'home_team': ['Arsenal', 'Man City', 'Real Madrid'],
            'away_team': ['Chelsea', 'Liverpool', 'Barcelona'],
            'outcome': ['home', 'home', 'draw'],
            'probability': [0.55, 0.60, 0.35],
            'odds': [2.20, 1.75, 3.20],
            'ev': [0.21, 0.05, 0.12],
            'stake': [50.0, 25.0, 30.0],
            'potential_profit': [60.0, 18.75, 66.0],
        })
    
    def test_generate_report_structure(self, sample_recommendations):
        """Test report generation structure."""
        report = generate_report(
            sample_recommendations,
            initial_bankroll=1000.0,
            ev_threshold=0.08,
            warnings=[]
        )
        
        assert 'timestamp' in report
        assert 'bankroll' in report
        assert 'summary' in report
        assert 'bets' in report
        assert 'warnings' in report
    
    def test_bankroll_tracking(self, sample_recommendations):
        """Test bankroll calculations."""
        report = generate_report(
            sample_recommendations,
            initial_bankroll=1000.0,
            ev_threshold=0.08,
            warnings=[]
        )
        
        # Total stake = 50 + 25 + 30 = 105
        assert report['bankroll']['initial'] == 1000.0
        assert report['bankroll']['used'] == 105.0
        assert report['bankroll']['remaining'] == 895.0
        assert report['bankroll']['utilization_pct'] == pytest.approx(10.5)
    
    def test_save_report_json(self, sample_recommendations):
        """Test JSON report saving."""
        from src.pipeline.reports import save_report_json
        
        report = generate_report(
            sample_recommendations,
            initial_bankroll=1000.0,
            ev_threshold=0.08,
            warnings=[]
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.json"
            save_report_json(report, output_path)
            
            assert output_path.exists()
            
            # Verify JSON is valid
            with open(output_path) as f:
                loaded = json.load(f)
            
            assert loaded['bankroll']['initial'] == 1000.0
    
    def test_save_report_txt(self, sample_recommendations):
        """Test text report saving."""
        from src.pipeline.reports import save_report_txt
        
        report = generate_report(
            sample_recommendations,
            initial_bankroll=1000.0,
            ev_threshold=0.08,
            warnings=['Test warning']
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.txt"
            save_report_txt(report, output_path)
            
            assert output_path.exists()
            
            # Verify content
            content = output_path.read_text()
            assert 'STAVKI BETTING RECOMMENDATIONS REPORT' in content
            assert 'BANKROLL STATUS' in content
            assert 'Test warning' in content


class TestWarnings:
    """Test warning collection."""
    
    def test_missing_odds_warning(self):
        """Test warning for missing odds."""
        features_df = pd.DataFrame({
            'match_id': [1, 2, 3],
            'home_team': ['A', 'B', 'C'],
        })
        
        odds_df = pd.DataFrame({
            'match_id': [1, 2],  # Missing ID 3
            'odds_home': [2.0, 2.0],
        })
        
        recommendations_df = pd.DataFrame()
        
        warnings = collect_warnings(features_df, odds_df, recommendations_df, 1000.0)
        
        assert any('missing odds' in w.lower() for w in warnings)
    
    def test_high_bankroll_usage_warning(self):
        """Test warning for high bankroll usage."""
        features_df = pd.DataFrame({'match_id': [1]})
        odds_df = pd.DataFrame({'match_id': [1]})
        
        recommendations_df = pd.DataFrame({
            'stake': [950.0]  # 95% of 1000
        })
        
        warnings = collect_warnings(features_df, odds_df, recommendations_df, 1000.0)
        
        assert any('high bankroll' in w.lower() for w in warnings)
    
    def test_no_bets_warning(self):
        """Test warning when no bets generated."""
        features_df = pd.DataFrame({'match_id': [1]})
        odds_df = pd.DataFrame({'match_id': [1]})
        recommendations_df = pd.DataFrame()  # Empty
        
        warnings = collect_warnings(features_df, odds_df, recommendations_df, 1000.0)
        
        assert any('no bets' in w.lower() for w in warnings)


class TestProductionCLI:
    """Test production CLI command."""
    
    def test_cli_with_ev_threshold(self):
        """Test CLI with EV threshold parameter."""
        runner = CliRunner()
        fixtures_dir = Path(__file__).parent / "fixtures"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                'run',
                '--matches', str(fixtures_dir / 'matches.csv'),
                '--odds', str(fixtures_dir / 'odds.csv'),
                '--bankroll', '1000',
                '--ev-threshold', '0.30',  # High threshold
                '--output', tmpdir
            ])
            
            assert result.exit_code == 0
            
            # Check files created
            output_dir = Path(tmpdir)
            assert (output_dir / 'bets.json').exists()
            assert (output_dir / 'bets.txt').exists()
    
    def test_cli_with_max_bets(self):
        """Test CLI with max bets limit."""
        runner = CliRunner()
        fixtures_dir = Path(__file__).parent / "fixtures"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                'run',
                '--matches', str(fixtures_dir / 'matches.csv'),
                '--odds', str(fixtures_dir / 'odds.csv'),
                '--bankroll', '1000',
                '--ev-threshold', '0.05',
                '--max-bets', '3',
                '--output', tmpdir
            ])
            
            assert result.exit_code == 0
            
            # Load report and check bet count
            output_dir = Path(tmpdir)
            with open(output_dir / 'bets.json') as f:
                report = json.load(f)
            
            # Should have at most 3 bets
            assert report['summary']['total_bets'] <= 3
    
    def test_output_file_structure(self):
        """Test that output files have correct structure."""
        runner = CliRunner()
        fixtures_dir = Path(__file__).parent / "fixtures"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                'run',
                '--matches', str(fixtures_dir / 'matches.csv'),
                '--odds', str(fixtures_dir / 'odds.csv'),
                '--bankroll', '1000',
                '--output', tmpdir
            ])
            
            assert result.exit_code == 0
            
            # Check JSON structure
            with open(Path(tmpdir) / 'bets.json') as f:
                report = json.load(f)
            
            assert 'bankroll' in report
            assert 'summary' in report
            assert 'bets' in report
            assert 'warnings' in report
            
            # Check required fields
            assert 'initial' in report['bankroll']
            assert 'used' in report['bankroll']
            assert 'remaining' in report['bankroll']


class TestStakeValidations:
    """Test stake validations in production mode."""
    
    def test_stakes_non_negative(self):
        """Test that all stakes are >= 0."""
        runner = CliRunner()
        fixtures_dir = Path(__file__).parent / "fixtures"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                'run',
                '--matches', str(fixtures_dir / 'matches.csv'),
                '--odds', str(fixtures_dir / 'odds.csv'),
                '--bankroll', '1000',
                '--output', tmpdir
            ])
            
            assert result.exit_code == 0
            
            with open(Path(tmpdir) / 'bets.json') as f:
                report = json.load(f)
            
            for bet in report['bets']:
                assert bet['stake'] >= 0
    
    def test_total_stake_within_bankroll(self):
        """Test total stake doesn't exceed bankroll."""
        runner = CliRunner()
        fixtures_dir = Path(__file__).parent / "fixtures"
        bankroll = 100.0  # Small bankroll
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                'run',
                '--matches', str(fixtures_dir / 'matches.csv'),
                '--odds', str(fixtures_dir / 'odds.csv'),
                '--bankroll', str(bankroll),
                '--output', tmpdir
            ])
            
            assert result.exit_code == 0
            
            with open(Path(tmpdir) / 'bets.json') as f:
                report = json.load(f)
            
            # Allow small margin for rounding
            assert report['summary']['total_stake'] <= bankroll * 1.01
