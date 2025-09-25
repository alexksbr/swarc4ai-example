import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import time

class TaxiFeatureStore:
    def __init__(self):
        self.df = None
        self.feature_cache = {}
        
    def load_data(self, parquet_path):
        """Load the parquet file"""
        self.df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(self.df)} trips")
        
    def compute_demand_features(self, zone_id, timestamp, print_performance=False, ttl_hours=1):
        """
        Calculate demand features for a zone at a specific time
        """
        start_total = time.time()
        cache_key = (zone_id, str(timestamp))

        if cache_key in self.feature_cache:
            print(f"Cache HIT for zone {zone_id} at {timestamp}")
            cached_time = self.feature_cache[cache_key].get('cached_at')
            age_hours = (datetime.now() - cached_time).total_seconds() / 3600
            
            if age_hours < ttl_hours:
                return self.feature_cache[cache_key]['features']

        print(f"Cache MISS for zone {zone_id} at {timestamp}")

        # Filter to the zone
        start_filter = time.time()
        zone_trips = self.df[self.df['PULocationID'] == zone_id]
        filter_time = time.time() - start_filter

        start_convert = time.time()
        if not pd.api.types.is_datetime64_any_dtype(zone_trips['tpep_pickup_datetime']):
            zone_trips = zone_trips.copy()
            zone_trips['tpep_pickup_datetime'] = pd.to_datetime(zone_trips['tpep_pickup_datetime'])
        convert_time = time.time() - start_convert

        start_feature_calculation = time.time()
        end_time = pd.to_datetime(timestamp)
        start_time_1h = end_time - timedelta(hours=1)
        start_time_3h = end_time - timedelta(hours=3)

        pickups_last_hour = zone_trips[
            (zone_trips['tpep_pickup_datetime'] > start_time_1h) &
            (zone_trips['tpep_pickup_datetime'] <= end_time)
        ].shape[0]

        pickups_last_3h = zone_trips[
            (zone_trips['tpep_pickup_datetime'] > start_time_3h) &
            (zone_trips['tpep_pickup_datetime'] <= end_time)
        ].shape[0]
        feature_calculation_time = time.time() - start_feature_calculation

        total_time = time.time() - start_total
        
        if print_performance:
            print(f"Performance breakdown:")
            print(f"  Filter to zone: {filter_time*1000:.2f}ms")
            print(f"  Datetime convert: {convert_time*1000:.2f}ms")
            print(f"  Feature calculation: {feature_calculation_time*1000:.2f}ms")
            print(f"  TOTAL: {total_time*1000:.2f}ms")

        entry = {
            "features": {
                'pickups_last_hour': pickups_last_hour,
                'pickups_last_3h': pickups_last_3h,
            }, 
            "cached_at": datetime.now()
        }

        
        # Store in cache for future use
        self.feature_cache[cache_key] = entry
        
        return entry["features"]

    def calculate_next_hour_pickups(self, zone_id, timestamp):
        """
        Calculate pickups in the NEXT hour from a given timestamp
        """
        # Filter to the zone
        zone_trips = self.df[self.df['PULocationID'] == zone_id]
        
        if not pd.api.types.is_datetime64_any_dtype(zone_trips['tpep_pickup_datetime']):
            zone_trips = zone_trips.copy()
            zone_trips['tpep_pickup_datetime'] = pd.to_datetime(zone_trips['tpep_pickup_datetime'])

        current_time = pd.to_datetime(timestamp)
        next_hour_end = current_time + timedelta(hours=1)

        pickups_next_hour = zone_trips[
            (zone_trips['tpep_pickup_datetime'] > current_time) &
            (zone_trips['tpep_pickup_datetime'] <= next_hour_end)
        ].shape[0]
        
        return pickups_next_hour

    def stress_test(self, n_requests=100):
        """
        Simulate multiple concurrent requests
        """
        zones = [161, 162, 237, 236]  # Times Square, Midtown, Upper East, Upper West
        start = time.time()
        
        for i in range(n_requests):
            zone = zones[i % 4]
            timestamp = f"2025-01-15 {(i % 24):02d}:00:00"
            features = self.compute_demand_features(zone, timestamp)
        
        total_time = time.time() - start
        avg_time = total_time / n_requests * 1000
        
        print(f"\nStress test results:")
        print(f"  Total requests: {n_requests}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average per request: {avg_time:.2f}ms")
        print(f"  Requests per second: {n_requests/total_time:.1f}")