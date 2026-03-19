"""
Logging configuration for the AI Agent application.
Provides structured logging with file rotation and console output.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "logs/app.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        format_string: Custom format string for log messages
    """
    # Remove default logger
    logger.remove()
    
    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Console output
    logger.add(
        sys.stdout,
        format=format_string,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # File output with rotation (loguru format)
    logger.add(
        log_file,
        format=format_string,
        level=log_level,
        rotation=max_bytes,
        retention=f"{backup_count} days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    logger.info(f"Logging initialized. Level: {log_level}, Log file: {log_file}")


def get_logger(name: str) -> logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Name for the logger (usually __name__)
    
    Returns:
        Configured logger instance
    """
    return logger.bind(name=name)


# Default logging setup
setup_logging()

__all__ = ['logger', 'setup_logging', 'get_logger']
