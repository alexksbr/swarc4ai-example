"""
Feature store for taxi demand prediction.

This module provides functionality for loading taxi trip data,
computing demand features, and caching results for performance.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_PARQUET_FILES, DEFAULT_CACHE_TTL_HOURS, REQUIRED_COLUMNS,
    LOOKBACK_HOURS_1, LOOKBACK_HOURS_3
)
from .utils import validate_zone_id, validate_hour

logger = logging.getLogger(__name__)


class TaxiFeatureStore:
    """
    Feature store for taxi demand prediction with caching capabilities.
    
    This class handles loading parquet data, computing demand features,
    and providing caching for improved performance.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the feature store.
        
        Args:
            config: Data configuration object. If None, uses default config.
        """
        # Default configuration
        default_config = {
            'data_dir': Path("data"),
            'parquet_files': DEFAULT_PARQUET_FILES,
            'cache_ttl_hours': DEFAULT_CACHE_TTL_HOURS
        }
        self.config = {**default_config, **(config or {})}
        self.df: Optional[pd.DataFrame] = None
        self.feature_cache: Dict[Tuple[int, str], Dict] = {}
        self._is_data_loaded = False
        
    def load_data(self, parquet_path: Union[str, Path]) -> None:
        """
        Load taxi trip data from a parquet file.
        
        Args:
            parquet_path: Path to the parquet file
            
        Raises:
            FileNotFoundError: If the parquet file doesn't exist
            ValueError: If the parquet file is invalid or empty
        """
        parquet_path = Path(parquet_path)
        
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        
        try:
            new_df = pd.read_parquet(parquet_path)
            
            if new_df.empty:
                raise ValueError(f"Parquet file is empty: {parquet_path}")
            
            # Validate required columns
            missing_columns = [col for col in REQUIRED_COLUMNS if col not in new_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Concatenate with existing data if any
            if self.df is not None:
                self.df = pd.concat([self.df, new_df], ignore_index=True)
            else:
                self.df = new_df
                
            self._is_data_loaded = True
            logger.info(f"Loaded {len(new_df)} trips from {parquet_path}")
            logger.info(f"Total trips in memory: {len(self.df)}")
            
        except Exception as e:
            logger.error(f"Error loading parquet file {parquet_path}: {e}")
            raise
    
    def load_all_data(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Load all configured parquet files.
        
        Args:
            data_dir: Directory containing parquet files. If None, uses config.
        """
        if data_dir is None:
            data_dir = self.config['data_dir']
        
        data_dir = Path(data_dir)
        
        for filename in self.config['parquet_files']:
            file_path = data_dir / filename
            try:
                self.load_data(file_path)
            except FileNotFoundError:
                logger.warning(f"Skipping missing file: {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                raise
    
    def compute_demand_features(
        self,
        zone_id: int,
        timestamp: Union[str, datetime],
        print_performance: bool = False,
        ttl_hours: Optional[float] = None
    ) -> Dict[str, int]:
        """
        Calculate demand features for a zone at a specific time.
        
        Args:
            zone_id: The pickup location zone ID
            timestamp: The timestamp for feature calculation
            print_performance: Whether to print performance metrics
            ttl_hours: Cache TTL in hours. If None, uses config default.
            
        Returns:
            Dictionary containing demand features
            
        Raises:
            ValueError: If data is not loaded or zone_id is invalid
        """
        if not self._is_data_loaded or self.df is None:
            raise ValueError("Data must be loaded before computing features")
        
        if not validate_zone_id(zone_id):
            raise ValueError(f"Invalid zone ID: {zone_id}")
        
        if ttl_hours is None:
            ttl_hours = self.config['cache_ttl_hours']
        
        start_total = time.time()
        cache_key = (zone_id, str(timestamp))

        # Check cache
        if cache_key in self.feature_cache:
            cached_entry = self.feature_cache[cache_key]
            cached_time = cached_entry.get('cached_at')
            if cached_time:
                age_hours = (datetime.now() - cached_time).total_seconds() / 3600
                
                if age_hours < ttl_hours:
                    if print_performance:
                        logger.info(f"Cache HIT for zone {zone_id} at {timestamp}")
                    return cached_entry['features']

        if print_performance:
            logger.info(f"Cache MISS for zone {zone_id} at {timestamp}")

        # Compute features
        try:
            features = self._compute_features_for_zone(
                zone_id, timestamp, print_performance
            )
            
            # Cache the result
            self.feature_cache[cache_key] = {
                "features": features,
                "cached_at": datetime.now()
            }
            
            total_time = time.time() - start_total
            if print_performance:
                logger.info(f"Total computation time: {total_time*1000:.2f}ms")
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing features for zone {zone_id} at {timestamp}: {e}")
            raise
    
    def _compute_features_for_zone(
        self,
        zone_id: int,
        timestamp: Union[str, datetime],
        print_performance: bool = False
    ) -> Dict[str, int]:
        """
        Internal method to compute features for a specific zone and time.
        
        Args:
            zone_id: The pickup location zone ID
            timestamp: The timestamp for feature calculation
            print_performance: Whether to print performance metrics
            
        Returns:
            Dictionary containing computed features
        """
        # Filter to the zone
        start_filter = time.time()
        zone_trips = self.df[self.df['PULocationID'] == zone_id].copy()
        filter_time = time.time() - start_filter

        # Convert datetime if needed
        start_convert = time.time()
        if not pd.api.types.is_datetime64_any_dtype(zone_trips['tpep_pickup_datetime']):
            zone_trips['tpep_pickup_datetime'] = pd.to_datetime(zone_trips['tpep_pickup_datetime'])
        convert_time = time.time() - start_convert

        # Calculate time windows
        start_feature_calculation = time.time()
        end_time = pd.to_datetime(timestamp)
        start_time_1h = end_time - timedelta(hours=LOOKBACK_HOURS_1)
        start_time_3h = end_time - timedelta(hours=LOOKBACK_HOURS_3)

        # Count pickups in different time windows
        pickups_last_hour = len(zone_trips[
            (zone_trips['tpep_pickup_datetime'] > start_time_1h) &
            (zone_trips['tpep_pickup_datetime'] <= end_time)
        ])

        pickups_last_3h = len(zone_trips[
            (zone_trips['tpep_pickup_datetime'] > start_time_3h) &
            (zone_trips['tpep_pickup_datetime'] <= end_time)
        ])
        
        feature_calculation_time = time.time() - start_feature_calculation

        if print_performance:
            logger.info(f"Performance breakdown:")
            logger.info(f"  Filter to zone: {filter_time*1000:.2f}ms")
            logger.info(f"  Datetime convert: {convert_time*1000:.2f}ms")
            logger.info(f"  Feature calculation: {feature_calculation_time*1000:.2f}ms")

        return {
            'pickups_last_hour': pickups_last_hour,
            'pickups_last_3h': pickups_last_3h,
        }

    def calculate_next_hour_pickups(
        self,
        zone_id: int,
        timestamp: Union[str, datetime]
    ) -> int:
        """
        Calculate pickups in the NEXT hour from a given timestamp.
        
        Args:
            zone_id: The pickup location zone ID
            timestamp: The starting timestamp
            
        Returns:
            Number of pickups in the next hour
            
        Raises:
            ValueError: If data is not loaded
        """
        if not self._is_data_loaded or self.df is None:
            raise ValueError("Data must be loaded before calculating pickups")
        
        if not validate_zone_id(zone_id):
            raise ValueError(f"Invalid zone ID: {zone_id}")
        
        try:
            # Filter to the zone
            zone_trips = self.df[self.df['PULocationID'] == zone_id].copy()
            
            # Ensure datetime column is properly formatted
            if not pd.api.types.is_datetime64_any_dtype(zone_trips['tpep_pickup_datetime']):
                zone_trips['tpep_pickup_datetime'] = pd.to_datetime(zone_trips['tpep_pickup_datetime'])

            current_time = pd.to_datetime(timestamp)
            next_hour_end = current_time + timedelta(hours=1)

            pickups_next_hour = len(zone_trips[
                (zone_trips['tpep_pickup_datetime'] > current_time) &
                (zone_trips['tpep_pickup_datetime'] <= next_hour_end)
            ])
            
            return pickups_next_hour
            
        except Exception as e:
            logger.error(f"Error calculating next hour pickups for zone {zone_id} at {timestamp}: {e}")
            raise

    def stress_test(self, n_requests: int = 100, zones: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Simulate multiple concurrent requests to test performance.
        
        Args:
            n_requests: Number of requests to simulate
            zones: List of zone IDs to test. If None, uses default zones.
            
        Returns:
            Dictionary with performance statistics
        """
        if zones is None:
            from .constants import DEFAULT_TEST_ZONES
            zones = DEFAULT_TEST_ZONES + [237, 236]  # Add Upper East, Upper West
        
        start = time.time()
        
        for i in range(n_requests):
            zone = zones[i % len(zones)]
            timestamp = f"2025-01-15 {(i % 24):02d}:00:00"
            try:
                self.compute_demand_features(zone, timestamp)
            except Exception as e:
                logger.error(f"Error in stress test request {i}: {e}")
        
        total_time = time.time() - start
        avg_time = total_time / n_requests * 1000
        
        stats = {
            "total_requests": n_requests,
            "total_time": total_time,
            "avg_time_ms": avg_time,
            "requests_per_second": n_requests / total_time if total_time > 0 else 0,
            "cache_size": len(self.feature_cache)
        }
        
        logger.info(f"Stress test results:")
        logger.info(f"  Total requests: {stats['total_requests']}")
        logger.info(f"  Total time: {stats['total_time']:.2f}s")
        logger.info(f"  Average per request: {stats['avg_time_ms']:.2f}ms")
        logger.info(f"  Requests per second: {stats['requests_per_second']:.1f}")
        logger.info(f"  Cache size: {stats['cache_size']}")
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self.feature_cache.clear()
        logger.info("Feature cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get statistics about the feature cache.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": len(self.feature_cache),
            "total_zones": len(set(key[0] for key in self.feature_cache.keys())),
        }
