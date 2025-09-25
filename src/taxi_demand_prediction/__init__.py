"""
Taxi demand prediction package.

A comprehensive system for predicting taxi pickup demand using historical data,
feature engineering, machine learning models, and real-time monitoring capabilities.
"""

from .feature_store import TaxiFeatureStore
from .model_serving import WaitTimePredictor
from .monitoring import DemandMonitor, AlertLevel, PredictionRecord, Alert
from .airport_predictor import AirportDemandPredictor
from .utils import setup_logging, validate_zone_id, validate_hour, calculate_mape
from .constants import (
    DEFAULT_TEST_ZONES, FEATURE_NAMES, MIN_ZONE_ID, MAX_ZONE_ID,
    DEFAULT_CACHE_TTL_HOURS, DEFAULT_N_ESTIMATORS
)

__version__ = "0.2.0"
__author__ = "Taxi Demand Prediction Team"

__all__ = [
    # Core classes
    'TaxiFeatureStore',
    'WaitTimePredictor', 
    'DemandMonitor',
    'AirportDemandPredictor',
    
    # Monitoring types
    'AlertLevel',
    'PredictionRecord',
    'Alert',
    
    # Utility functions
    'setup_logging',
    'validate_zone_id',
    'validate_hour',
    'calculate_mape',
    
    # Constants
    'DEFAULT_TEST_ZONES',
    'FEATURE_NAMES',
    'MIN_ZONE_ID',
    'MAX_ZONE_ID',
    'DEFAULT_CACHE_TTL_HOURS',
    'DEFAULT_N_ESTIMATORS',
    
    # Package metadata
    '__version__',
    '__author__'
]
