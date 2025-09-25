"""
Configuration management for the taxi demand prediction system.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    """Configuration for data-related settings."""
    data_dir: Path = Path("data")
    parquet_files: List[str] = None
    cache_ttl_hours: float = 1.0
    
    def __post_init__(self):
        if self.parquet_files is None:
            self.parquet_files = [
                "yellow_tripdata_2025-01.parquet",
                "yellow_tripdata_2025-02.parquet",
                "yellow_tripdata_2025-03.parquet",
            ]


@dataclass
class ModelConfig:
    """Configuration for model-related settings."""
    n_estimators: int = 100
    random_state: int = 42
    test_zones: List[int] = None
    
    def __post_init__(self):
        if self.test_zones is None:
            self.test_zones = [161, 162]  # Times Square, Midtown


@dataclass
class MonitoringConfig:
    """Configuration for monitoring settings."""
    alert_threshold_mape: float = 30.0  # 30% MAPE threshold
    drift_window_size: int = 20
    min_samples_for_drift: int = 10


@dataclass
class AppConfig:
    """Main application configuration."""
    data: DataConfig = None
    model: ModelConfig = None
    monitoring: MonitoringConfig = None
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()


def get_config() -> AppConfig:
    """Get the application configuration."""
    return AppConfig()
