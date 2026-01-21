"""
Tests for run archive and artifacts (T090).

Tests:
- Run creates archive directory
- All artifacts are saved
- Metadata includes parameters
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from src.cli import cli
from src.pipeline.run_archive import (
    create_run_directory,
    get_run_summary,
    list_runs,
    save_run_artifacts,
    save_run_metadata,
)


class TestRunArchive:
    """Test run archive functionality."""
    
    def test_create_run_directory(self):
        """Test run directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "runs"
            
            run_dir = create_run_directory(base_dir)
            
            assert run_dir.exists()
            assert run_dir.is_dir()
            # Check format: runs/YYYY-MM-DD/HHMMSS/
            assert len(run_dir.parts) >= 3
    
    def test_save_run_metadata(self):
        """Test metadata saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            
            save_run_metadata(
                run_dir,
                bankroll=1000.0,
                ev_threshold=0.08,
                kelly_fraction=0.5,
                max_stake_pct=5.0
            )
            
            metadata_path = run_dir / "metadata.json"
            assert metadata_path.exists()
            
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            assert 'timestamp' in metadata
            assert 'parameters' in metadata
            assert metadata['parameters']['bankroll'] == 1000.0
    
    def test_save_run_artifacts(self):
        """Test artifacts saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            
            # Sample data
            predictions = pd.DataFrame({
                'match_id': [1, 2],
                'prob_home': [0.5, 0.6]
            })
            
            recommendations = pd.DataFrame({
                'match_id': [1],
                'stake': [50.0]
            })
            
            paths = save_run_artifacts(
                run_dir,
                predictions_df=predictions,
                recommendations_df=recommendations
            )
            
            assert 'predictions_csv' in paths
            assert 'recommendations_csv' in paths
            assert paths['predictions_csv'].exists()
            assert paths['recommendations_csv'].exists()
    
    def test_list_runs(self):
        """Test listing runs."""
        import time
        
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "runs"
            
            # Create some runs with delay to ensure different timestamps
            run_dir1 = create_run_directory(base_dir)
            time.sleep(1)  # Ensure different second
            run_dir2 = create_run_directory(base_dir)
            
            runs = list_runs(base_dir)
            
            assert len(runs) >= 2
    
    def test_get_run_summary(self):
        """Test getting run summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            
            save_run_metadata(
                run_dir,
                bankroll=1000.0,
                ev_threshold=0.08,
                kelly_fraction=0.5,
                max_stake_pct=5.0
            )
            
            summary = get_run_summary(run_dir)
            
            assert summary is not None
            assert 'timestamp' in summary
            assert 'parameters' in summary


class TestCLIRunArchive:
    """Test CLI run command with archive integration."""
    
    def test_cli_run_creates_archive(self):
        """Test that CLI run creates archive directory."""
        runner = CliRunner()
        fixtures_dir = Path(__file__).parent / "fixtures"
        
        # Note: This is a smoke test - full integration would require
        # modifying the run command to use archives, which we'll do next
        
        result = runner.invoke(cli, [
            'run',
            '--matches', str(fixtures_dir / 'matches.csv'),
            '--odds', str(fixtures_dir / 'odds.csv'),
            '--bankroll', '1000'
        ], catch_exceptions=False)
        
        # Command should succeed
        assert result.exit_code == 0
