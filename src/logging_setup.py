"""
Centralized logging setup for STAVKI system.
Provides structured logging with proper formatting and rotation.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import structlog
from colorama import Fore, Style, init as colorama_init


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    enable_colors: bool = True
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (creates parent directories if needed)
        enable_colors: Enable colored console output
    """
    # Initialize colorama for Windows compatibility
    if enable_colors:
        colorama_init(autoreset=True)
    
    # Ensure log directory exists
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure stdlib logging
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        handlers=[]
    )
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if enable_colors:
        console_formatter = ColoredFormatter()
    else:
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    console_handler.setFormatter(console_formatter)
    
    # File handler (no colors)
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers = []
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Log initial message
    logger = get_logger(__name__)
    logger.info(
        "Logging initialized",
        log_level=log_level,
        log_file=str(log_file) if log_file else None
    )


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, "")
        reset = Style.RESET_ALL if color else ""
        
        # Format timestamp
        timestamp = self.formatTime(record, "%H:%M:%S")
        
        # Format message
        message = record.getMessage()
        
        # Build colored output
        return f"{Fore.WHITE}{timestamp}{reset} {color}[{record.levelname:8}]{reset} {Fore.BLUE}{record.name}{reset}: {message}"


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger
    """
    return structlog.get_logger(name)


# Convenience function for module-level logging
def log_step(logger: structlog.BoundLogger, step: str, **kwargs) -> None:
    """
    Log a processing step with context.
    
    Args:
        logger: Logger instance
        step: Description of the step
        **kwargs: Additional context to log
    """
    logger.info(f"→ {step}", **kwargs)
