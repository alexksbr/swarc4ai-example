"""
Utility functions for the taxi demand prediction system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None for console only)
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[]
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format_string))
    logging.getLogger().addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logging.getLogger().addHandler(file_handler)
        
        logging.info(f"Logging to file: {log_file}")
    
    logging.info(f"Logging configured at {level} level")


def validate_zone_id(zone_id: int) -> bool:
    """
    Validate if a zone ID is reasonable for NYC taxi data.
    
    Args:
        zone_id: Zone ID to validate
        
    Returns:
        True if zone ID appears valid, False otherwise
    """
    # NYC taxi zones are typically between 1 and 265
    return isinstance(zone_id, int) and 1 <= zone_id <= 265


def validate_hour(hour: int) -> bool:
    """
    Validate if an hour value is valid (0-23).
    
    Args:
        hour: Hour to validate
        
    Returns:
        True if hour is valid, False otherwise
    """
    return isinstance(hour, int) and 0 <= hour <= 23
