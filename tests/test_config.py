"""
Unit tests for configuration module.
"""

import pytest
from pathlib import Path
import tempfile
import os

from src.config import Config


class TestConfig:
    """Test cases for Config class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        
        assert config.dry_run is True
        assert config.log_level == "INFO"
        assert config.min_ev_threshold == 0.08
        assert config.kelly_fraction == 0.25
        assert config.max_stake_percent == 5.0
        assert config.initial_bankroll == 1000.0
    
    def test_config_validation_valid(self):
        """Test validation with valid parameters."""
        config = Config()
        config.validate()  # Should not raise
    
    def test_config_validation_invalid_ev(self):
        """Test validation with invalid EV threshold."""
        config = Config(min_ev_threshold=-0.1)
        with pytest.raises(ValueError, match="min_ev_threshold"):
            config.validate()
    
    def test_config_validation_invalid_kelly(self):
        """Test validation with invalid Kelly fraction."""
        config = Config(kelly_fraction=1.5)
        with pytest.raises(ValueError, match="kelly_fraction"):
            config.validate()
    
    def test_config_validation_invalid_stake(self):
        """Test validation with invalid max stake."""
        config = Config(max_stake_percent=150.0)
        with pytest.raises(ValueError, match="max_stake_percent"):
            config.validate()
    
    def test_config_from_env(self):
        """Test loading configuration from environment."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("DRY_RUN=false\n")
            f.write("LOG_LEVEL=DEBUG\n")
            f.write("MIN_EV_THRESHOLD=0.10\n")
            f.write("INITIAL_BANKROLL=5000.0\n")
            env_file = Path(f.name)
        
        try:
            config = Config.from_env(env_file)
            
            assert config.dry_run is False
            assert config.log_level == "DEBUG"
            assert config.min_ev_threshold == 0.10
            assert config.initial_bankroll == 5000.0
        finally:
            os.unlink(env_file)
    
    def test_config_create_directories(self, tmp_path):
        """Test directory creation."""
        config = Config(
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
            outputs_dir=tmp_path / "outputs",
            logs_dir=tmp_path / "logs"
        )
        
        config.create_directories()
        
        assert config.data_dir.exists()
        assert config.models_dir.exists()
        assert config.outputs_dir.exists()
        assert config.logs_dir.exists()
    
    def test_config_str_representation(self):
        """Test string representation of config."""
        config = Config()
        config_str = str(config)
        
        assert "STAVKI Configuration" in config_str
        assert "DRY RUN" in config_str
        assert "8.0%" in config_str  # min_ev_threshold
