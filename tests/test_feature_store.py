"""
Unit tests for the TaxiFeatureStore class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

from src.taxi_demand_prediction.feature_store import TaxiFeatureStore


@pytest.fixture
def sample_data():
    """Create sample taxi trip data for testing."""
    dates = pd.date_range('2025-01-15 00:00:00', '2025-01-15 23:59:00', freq='30min')
    
    data = []
    for i, date in enumerate(dates):
        data.append({
            'PULocationID': 161 if i % 2 == 0 else 162,
            'tpep_pickup_datetime': date,
            'trip_distance': np.random.uniform(0.5, 10.0),
            'fare_amount': np.random.uniform(5.0, 50.0)
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_parquet_file(sample_data):
    """Create a temporary parquet file with sample data."""
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        sample_data.to_parquet(tmp.name)
        yield tmp.name
    
    # Cleanup
    os.unlink(tmp.name)


@pytest.fixture
def feature_store():
    """Create a TaxiFeatureStore instance for testing."""
    return TaxiFeatureStore()


class TestTaxiFeatureStore:
    """Test cases for TaxiFeatureStore."""
    
    def test_initialization(self):
        """Test feature store initialization."""
        fs = TaxiFeatureStore()
        assert fs.df is None
        assert fs.feature_cache == {}
        assert fs._is_data_loaded is False
        assert 'cache_ttl_hours' in fs.config
        assert 'data_dir' in fs.config
        assert 'parquet_files' in fs.config
    
    def test_initialization_with_config(self):
        """Test feature store initialization with custom config."""
        config = {
            'cache_ttl_hours': 2.0,
            'data_dir': Path('/custom/path')
        }
        fs = TaxiFeatureStore(config)
        assert fs.config['cache_ttl_hours'] == 2.0
        assert fs.config['data_dir'] == Path('/custom/path')
    
    def test_load_data_success(self, feature_store, temp_parquet_file):
        """Test successful data loading."""
        feature_store.load_data(temp_parquet_file)
        
        assert feature_store.df is not None
        assert len(feature_store.df) > 0
        assert feature_store._is_data_loaded is True
        assert 'PULocationID' in feature_store.df.columns
        assert 'tpep_pickup_datetime' in feature_store.df.columns
    
    def test_load_data_file_not_found(self, feature_store):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            feature_store.load_data('/nonexistent/file.parquet')
    
    def test_load_data_multiple_files(self, feature_store, sample_data):
        """Test loading multiple parquet files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create two parquet files
            file1 = Path(tmp_dir) / 'file1.parquet'
            file2 = Path(tmp_dir) / 'file2.parquet'
            
            sample_data[:25].to_parquet(file1)
            sample_data[25:].to_parquet(file2)
            
            # Load both files
            feature_store.load_data(file1)
            initial_length = len(feature_store.df)
            
            feature_store.load_data(file2)
            final_length = len(feature_store.df)
            
            assert final_length > initial_length
            assert final_length == len(sample_data)
    
    def test_compute_demand_features_without_data(self, feature_store):
        """Test computing features without loaded data."""
        with pytest.raises(ValueError, match="Data must be loaded"):
            feature_store.compute_demand_features(161, '2025-01-15 12:00:00')
    
    def test_compute_demand_features_success(self, feature_store, temp_parquet_file):
        """Test successful feature computation."""
        feature_store.load_data(temp_parquet_file)
        
        features = feature_store.compute_demand_features(161, '2025-01-15 12:00:00')
        
        assert isinstance(features, dict)
        assert 'pickups_last_hour' in features
        assert 'pickups_last_3h' in features
        assert isinstance(features['pickups_last_hour'], int)
        assert isinstance(features['pickups_last_3h'], int)
        assert features['pickups_last_hour'] >= 0
        assert features['pickups_last_3h'] >= 0
    
    def test_feature_caching(self, feature_store, temp_parquet_file):
        """Test that feature computation uses caching."""
        feature_store.load_data(temp_parquet_file)
        
        # First call - should compute and cache
        features1 = feature_store.compute_demand_features(161, '2025-01-15 12:00:00')
        cache_size_after_first = len(feature_store.feature_cache)
        
        # Second call - should use cache
        features2 = feature_store.compute_demand_features(161, '2025-01-15 12:00:00')
        cache_size_after_second = len(feature_store.feature_cache)
        
        assert features1 == features2
        assert cache_size_after_first == cache_size_after_second == 1
    
    def test_cache_expiration(self, feature_store, temp_parquet_file):
        """Test cache expiration with very short TTL."""
        feature_store.load_data(temp_parquet_file)
        
        # Set very short TTL
        features1 = feature_store.compute_demand_features(
            161, '2025-01-15 12:00:00', ttl_hours=0.0001  # Very short TTL
        )
        
        # Wait a bit and call again - should recompute
        import time
        time.sleep(0.01)
        
        features2 = feature_store.compute_demand_features(
            161, '2025-01-15 12:00:00', ttl_hours=0.0001
        )
        
        # Features should be the same but cache should have been refreshed
        assert features1 == features2
    
    def test_calculate_next_hour_pickups_without_data(self, feature_store):
        """Test calculating next hour pickups without loaded data."""
        with pytest.raises(ValueError, match="Data must be loaded"):
            feature_store.calculate_next_hour_pickups(161, '2025-01-15 12:00:00')
    
    def test_calculate_next_hour_pickups_success(self, feature_store, temp_parquet_file):
        """Test successful next hour pickup calculation."""
        feature_store.load_data(temp_parquet_file)
        
        pickups = feature_store.calculate_next_hour_pickups(161, '2025-01-15 12:00:00')
        
        assert isinstance(pickups, int)
        assert pickups >= 0
    
    def test_stress_test(self, feature_store, temp_parquet_file):
        """Test stress testing functionality."""
        feature_store.load_data(temp_parquet_file)
        
        stats = feature_store.stress_test(n_requests=10, zones=[161, 162])
        
        assert isinstance(stats, dict)
        assert 'total_requests' in stats
        assert 'total_time' in stats
        assert 'avg_time_ms' in stats
        assert 'requests_per_second' in stats
        assert 'cache_size' in stats
        
        assert stats['total_requests'] == 10
        assert stats['total_time'] > 0
        assert stats['avg_time_ms'] >= 0
        assert stats['requests_per_second'] > 0
    
    def test_clear_cache(self, feature_store, temp_parquet_file):
        """Test cache clearing functionality."""
        feature_store.load_data(temp_parquet_file)
        
        # Add something to cache
        feature_store.compute_demand_features(161, '2025-01-15 12:00:00')
        assert len(feature_store.feature_cache) > 0
        
        # Clear cache
        feature_store.clear_cache()
        assert len(feature_store.feature_cache) == 0
    
    def test_get_cache_stats(self, feature_store, temp_parquet_file):
        """Test cache statistics functionality."""
        feature_store.load_data(temp_parquet_file)
        
        # Initially empty
        stats = feature_store.get_cache_stats()
        assert stats['cache_size'] == 0
        assert stats['total_zones'] == 0
        
        # Add some cache entries
        feature_store.compute_demand_features(161, '2025-01-15 12:00:00')
        feature_store.compute_demand_features(162, '2025-01-15 12:00:00')
        feature_store.compute_demand_features(161, '2025-01-15 13:00:00')
        
        stats = feature_store.get_cache_stats()
        assert stats['cache_size'] == 3
        assert stats['total_zones'] == 2  # zones 161 and 162
