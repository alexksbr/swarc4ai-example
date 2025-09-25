"""
Constants for the taxi demand prediction system.
"""

from typing import List

# NYC Taxi Zone Constants
MIN_ZONE_ID = 1
MAX_ZONE_ID = 265
DEFAULT_TEST_ZONES = [161, 162]  # Times Square, Midtown

# Time Constants
HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60

# Feature Engineering Constants
DEFAULT_CACHE_TTL_HOURS = 1.0
LOOKBACK_HOURS_1 = 1
LOOKBACK_HOURS_3 = 3

# Model Constants
DEFAULT_N_ESTIMATORS = 100
DEFAULT_RANDOM_STATE = 42
EPSILON_FOR_MAPE = 1e-8

# Monitoring Constants
DEFAULT_ALERT_THRESHOLD_MAPE = 30.0
DEFAULT_DRIFT_WINDOW_SIZE = 20
DEFAULT_MIN_SAMPLES_FOR_DRIFT = 10
DEFAULT_PERFORMANCE_WINDOW_HOURS = 24
DEFAULT_ALERT_COOLDOWN_MINUTES = 60
DEFAULT_DATA_RETENTION_DAYS = 7

# Revenue Constants
CITY_AVG_FARE = 15.50
CITY_AVG_TRIP_TIME_MINUTES = 12
AIRPORT_AVG_FARE = 45.00
AIRPORT_AVG_TRIP_TIME_MINUTES = 35

# Airport Demand Constants
AIRPORT_PASSENGER_TAXI_RATE = 0.3
INTERNATIONAL_FLIGHT_MULTIPLIER = 1.2
DEMAND_NORMALIZATION_CITY = 500.0
DEMAND_NORMALIZATION_AIRPORT = 200.0

# File Constants
DEFAULT_PARQUET_FILES: List[str] = [
    "yellow_tripdata_2025-01.parquet",
    "yellow_tripdata_2025-02.parquet",
    "yellow_tripdata_2025-03.parquet",
]

# Required DataFrame Columns
REQUIRED_COLUMNS: List[str] = ['PULocationID', 'tpep_pickup_datetime']

# Feature Names
FEATURE_NAMES: List[str] = ['zone_id', 'hour', 'pickups_last_hour', 'pickups_last_3h']

# Airport Zones (example - these would be actual JFK/LGA/EWR zone IDs)
AIRPORT_ZONES: List[int] = [132, 138, 161]  # Example airport zone IDs

# Peak Hours
PEAK_MORNING_START = 6
PEAK_MORNING_END = 10
PEAK_EVENING_START = 17
PEAK_EVENING_END = 20

# Flight Schedule Constants
DAYTIME_FLIGHT_RANGE = (3, 8)
NIGHTTIME_FLIGHT_RANGE = (0, 2)
DAYTIME_HOURS = (6, 22)
FLIGHT_ARRIVAL_DELAY_RANGE = (30, 90)  # minutes
PASSENGER_RANGE = (100, 300)
INTERNATIONAL_FLIGHT_PROBABILITY = 0.3
