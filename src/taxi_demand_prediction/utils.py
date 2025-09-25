"""
Utility functions for the taxi demand prediction system.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .constants import (
    MIN_ZONE_ID, MAX_ZONE_ID, HOURS_PER_DAY, EPSILON_FOR_MAPE
)


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
    return isinstance(zone_id, int) and MIN_ZONE_ID <= zone_id <= MAX_ZONE_ID


def validate_hour(hour: int) -> bool:
    """
    Validate if an hour value is valid (0-23).
    
    Args:
        hour: Hour to validate
        
    Returns:
        True if hour is valid, False otherwise
    """
    return isinstance(hour, int) and 0 <= hour < HOURS_PER_DAY


def calculate_mape(
    actuals: List[Union[int, float]], 
    predictions: List[float]
) -> float:
    """
    Calculate Mean Absolute Percentage Error with proper zero handling.
    
    This is a shared utility function to avoid code duplication across modules.
    
    Args:
        actuals: Actual values
        predictions: Predicted values
        
    Returns:
        MAPE as a percentage
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not actuals or not predictions:
        raise ValueError("Actuals and predictions cannot be empty")
    
    if len(actuals) != len(predictions):
        raise ValueError("Actuals and predictions must have the same length")
    
    percentage_errors = []
    
    for pred, actual in zip(predictions, actuals):
        if actual == 0 and pred == 0:
            # Both are zero, perfect prediction
            percentage_errors.append(0.0)
        elif actual == 0:
            # Actual is zero but prediction is not, use absolute error
            percentage_errors.append(abs(pred))
        else:
            # Standard MAPE calculation
            percentage_error = abs((actual - pred) / (actual + EPSILON_FOR_MAPE)) * 100
            percentage_errors.append(percentage_error)
    
    return float(np.mean(percentage_errors))


def validate_config_dict(config: dict, required_keys: List[str]) -> None:
    """
    Validate that a configuration dictionary contains required keys.
    
    Args:
        config: Configuration dictionary to validate
        required_keys: List of required keys
        
    Raises:
        ValueError: If required keys are missing
    """
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary")
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if denominator is zero
        
    Returns:
        Division result or default value
    """
    if abs(denominator) < EPSILON_FOR_MAPE:
        return default
    return numerator / denominator


def ensure_non_negative(value: Union[int, float]) -> Union[int, float]:
    """
    Ensure a value is non-negative.
    
    Args:
        value: Input value
        
    Returns:
        Maximum of value and 0
    """
    return max(0, value)
