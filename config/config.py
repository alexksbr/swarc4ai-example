"""
Configuration management for the taxi demand prediction system.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.taxi_demand_prediction.constants import (
    DEFAULT_PARQUET_FILES, DEFAULT_CACHE_TTL_HOURS, DEFAULT_N_ESTIMATORS,
    DEFAULT_RANDOM_STATE, DEFAULT_TEST_ZONES, DEFAULT_ALERT_THRESHOLD_MAPE,
    DEFAULT_DRIFT_WINDOW_SIZE, DEFAULT_MIN_SAMPLES_FOR_DRIFT
)


@dataclass
class DataConfig:
    """Configuration for data-related settings."""
    data_dir: Path = field(default_factory=lambda: Path("data"))
    parquet_files: List[str] = field(default_factory=lambda: DEFAULT_PARQUET_FILES.copy())
    cache_ttl_hours: float = DEFAULT_CACHE_TTL_HOURS
    
    def __post_init__(self) -> None:
        """Post-initialization validation."""
        if not isinstance(self.data_dir, Path):
            self.data_dir = Path(self.data_dir)
        
        if not self.parquet_files:
            self.parquet_files = DEFAULT_PARQUET_FILES.copy()
        
        # Validate cache TTL
        if self.cache_ttl_hours <= 0:
            raise ValueError("cache_ttl_hours must be positive")


@dataclass
class ModelConfig:
    """Configuration for model-related settings."""
    n_estimators: int = DEFAULT_N_ESTIMATORS
    random_state: int = DEFAULT_RANDOM_STATE
    test_zones: List[int] = field(default_factory=lambda: DEFAULT_TEST_ZONES.copy())
    
    def __post_init__(self) -> None:
        """Post-initialization validation."""
        if not self.test_zones:
            self.test_zones = DEFAULT_TEST_ZONES.copy()
        
        # Validate model parameters
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        
        if not isinstance(self.random_state, int):
            raise ValueError("random_state must be an integer")
        
        # Validate zone IDs
        from src.taxi_demand_prediction.utils import validate_zone_id
        invalid_zones = [zone for zone in self.test_zones if not validate_zone_id(zone)]
        if invalid_zones:
            raise ValueError(f"Invalid zone IDs: {invalid_zones}")


@dataclass
class MonitoringConfig:
    """Configuration for monitoring settings."""
    alert_threshold_mape: float = DEFAULT_ALERT_THRESHOLD_MAPE
    drift_window_size: int = DEFAULT_DRIFT_WINDOW_SIZE
    min_samples_for_drift: int = DEFAULT_MIN_SAMPLES_FOR_DRIFT
    
    def __post_init__(self) -> None:
        """Post-initialization validation."""
        if self.alert_threshold_mape <= 0:
            raise ValueError("alert_threshold_mape must be positive")
        
        if self.drift_window_size <= 0:
            raise ValueError("drift_window_size must be positive")
        
        if self.min_samples_for_drift <= 0:
            raise ValueError("min_samples_for_drift must be positive")
        
        if self.min_samples_for_drift > self.drift_window_size:
            raise ValueError("min_samples_for_drift cannot exceed drift_window_size")


@dataclass
class AppConfig:
    """Main application configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    log_level: str = "INFO"
    
    def __post_init__(self) -> None:
        """Post-initialization validation."""
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"log_level must be one of {valid_log_levels}")
        
        # Ensure all sub-configs are initialized
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
    
    def validate(self) -> None:
        """Validate the entire configuration."""
        # Check if data directory exists
        if not self.data.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data.data_dir}")
        
        # Check if at least one parquet file exists
        existing_files = []
        for filename in self.data.parquet_files:
            file_path = self.data.data_dir / filename
            if file_path.exists():
                existing_files.append(filename)
        
        if not existing_files:
            raise FileNotFoundError(
                f"No parquet files found in {self.data.data_dir}. "
                f"Expected files: {self.data.parquet_files}"
            )


def get_config(
    data_dir: Optional[Path] = None,
    log_level: Optional[str] = None,
    **kwargs
) -> AppConfig:
    """
    Get the application configuration with optional overrides.
    
    Args:
        data_dir: Override for data directory
        log_level: Override for log level
        **kwargs: Additional configuration overrides
        
    Returns:
        Configured AppConfig instance
    """
    config_kwargs = {}
    
    # Handle data config overrides
    if data_dir is not None:
        config_kwargs['data'] = DataConfig(data_dir=data_dir)
    
    # Handle log level override
    if log_level is not None:
        config_kwargs['log_level'] = log_level
    
    # Apply any additional overrides
    config_kwargs.update(kwargs)
    
    return AppConfig(**config_kwargs)


def get_config_from_env() -> AppConfig:
    """
    Get configuration from environment variables.
    
    Environment variables:
        TAXI_DATA_DIR: Data directory path
        TAXI_LOG_LEVEL: Logging level
        TAXI_CACHE_TTL: Cache TTL in hours
        TAXI_N_ESTIMATORS: Number of estimators for model
        
    Returns:
        AppConfig instance with environment overrides
    """
    data_config = DataConfig()
    model_config = ModelConfig()
    monitoring_config = MonitoringConfig()
    
    # Override from environment
    if 'TAXI_DATA_DIR' in os.environ:
        data_config.data_dir = Path(os.environ['TAXI_DATA_DIR'])
    
    if 'TAXI_CACHE_TTL' in os.environ:
        data_config.cache_ttl_hours = float(os.environ['TAXI_CACHE_TTL'])
    
    if 'TAXI_N_ESTIMATORS' in os.environ:
        model_config.n_estimators = int(os.environ['TAXI_N_ESTIMATORS'])
    
    if 'TAXI_ALERT_THRESHOLD' in os.environ:
        monitoring_config.alert_threshold_mape = float(os.environ['TAXI_ALERT_THRESHOLD'])
    
    log_level = os.environ.get('TAXI_LOG_LEVEL', 'INFO')
    
    return AppConfig(
        data=data_config,
        model=model_config,
        monitoring=monitoring_config,
        log_level=log_level
    )
