"""
Configuration management using dataclasses and environment variables.
All configuration is type-safe and validated.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    """Main configuration class for STAVKI system."""
    
    # Application mode
    dry_run: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("logs/app.log")
    
    # API Keys (all optional, loaded from .env)
    betfair_api_key: Optional[str] = None
    pinnacle_api_key: Optional[str] = None
    bet365_api_key: Optional[str] = None
    odds_api_key: Optional[str] = None
    twitter_bearer_token: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    
    # Model parameters
    min_ev_threshold: float = 0.08  # 8% minimum edge
    kelly_fraction: float = 0.25    # Conservative Kelly
    max_stake_percent: float = 5.0  # Max 5% bankroll per bet
    
    # Bankroll management
    initial_bankroll: float = 1000.0
    bankroll_currency: str = "EUR"
    
    # Risk management
    max_daily_loss_percent: float = 10.0
    max_drawdown_percent: float = 20.0
    
    # Directories
    data_dir: Path = Path("data/")
    models_dir: Path = Path("models/")
    outputs_dir: Path = Path("outputs/")
    logs_dir: Path = Path("logs/")
    
    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "Config":
        """
        Load configuration from environment variables.
        
        Args:
            env_file: Path to .env file (default: .env in project root)
            
        Returns:
            Configured Config instance
        """
        # Load .env file if it exists
        if env_file is None:
            env_file = Path(".env")
        
        if env_file.exists():
            load_dotenv(env_file)
        
        # Helper to get env var with type conversion
        def get_env(key: str, default: any, converter: type = str) -> any:
            value = os.getenv(key)
            if value is None:
                return default
            try:
                if converter == bool:
                    return value.lower() in ("true", "1", "yes")
                elif converter == Path:
                    return Path(value)
                else:
                    return converter(value)
            except (ValueError, TypeError):
                return default
        
        return cls(
            # Application mode
            dry_run=get_env("DRY_RUN", True, bool),
            
            # Logging
            log_level=get_env("LOG_LEVEL", "INFO"),
            log_file=get_env("LOG_FILE", Path("logs/app.log"), Path),
            
            # API Keys
            betfair_api_key=get_env("BETFAIR_API_KEY", None),
            pinnacle_api_key=get_env("PINNACLE_API_KEY", None),
            bet365_api_key=get_env("BET365_API_KEY", None),
            odds_api_key=get_env("ODDS_API_KEY", None),
            twitter_bearer_token=get_env("TWITTER_BEARER_TOKEN", None),
            telegram_bot_token=get_env("TELEGRAM_BOT_TOKEN", None),
            telegram_chat_id=get_env("TELEGRAM_CHAT_ID", None),
            
            # Model parameters
            min_ev_threshold=get_env("MIN_EV_THRESHOLD", 0.08, float),
            kelly_fraction=get_env("KELLY_FRACTION", 0.25, float),
            max_stake_percent=get_env("MAX_STAKE_PERCENT", 5.0, float),
            
            # Bankroll
            initial_bankroll=get_env("INITIAL_BANKROLL", 1000.0, float),
            bankroll_currency=get_env("BANKROLL_CURRENCY", "EUR"),
            
            # Risk
            max_daily_loss_percent=get_env("MAX_DAILY_LOSS_PERCENT", 10.0, float),
            max_drawdown_percent=get_env("MAX_DRAWDOWN_PERCENT", 20.0, float),
            
            # Directories
            data_dir=get_env("DATA_DIR", Path("data/"), Path),
            models_dir=get_env("MODELS_DIR", Path("models/"), Path),
            outputs_dir=get_env("OUTPUTS_DIR", Path("outputs/"), Path),
        )
    
    def create_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for directory in [self.data_dir, self.models_dir, self.outputs_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if self.min_ev_threshold < 0:
            raise ValueError("min_ev_threshold must be non-negative")
        
        if not 0 < self.kelly_fraction <= 1:
            raise ValueError("kelly_fraction must be between 0 and 1")
        
        if not 0 < self.max_stake_percent <= 100:
            raise ValueError("max_stake_percent must be between 0 and 100")
        
        if self.initial_bankroll <= 0:
            raise ValueError("initial_bankroll must be positive")
        
        if not 0 <= self.max_daily_loss_percent <= 100:
            raise ValueError("max_daily_loss_percent must be between 0 and 100")
        
        if not 0 <= self.max_drawdown_percent <= 100:
            raise ValueError("max_drawdown_percent must be between 0 and 100")
    
    def __str__(self) -> str:
        """String representation (masks sensitive data)."""
        lines = ["STAVKI Configuration:"]
        lines.append(f"  Mode: {'DRY RUN' if self.dry_run else 'LIVE (REAL BETS!)'}")
        lines.append(f"  Log Level: {self.log_level}")
        lines.append(f"  Min EV Threshold: {self.min_ev_threshold:.1%}")
        lines.append(f"  Kelly Fraction: {self.kelly_fraction:.2f}")
        lines.append(f"  Max Stake: {self.max_stake_percent:.1f}%")
        lines.append(f"  Initial Bankroll: {self.initial_bankroll:.2f} {self.bankroll_currency}")
        return "\n".join(lines)
