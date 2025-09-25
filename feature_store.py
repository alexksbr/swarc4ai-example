import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import time

class TaxiFeatureStore:
    def __init__(self):
        self.df = None
        
    def load_data(self, parquet_path):
        """Load the parquet file"""
        self.df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(self.df)} trips")
        
    def compute_demand_features(self, zone_id, timestamp, print_performance=False):
        """
        Calculate demand features for a zone at a specific time
        """
        start_total = time.time()

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

        return {
            'pickups_last_hour': pickups_last_hour,
            'pickups_last_3h': pickups_last_3h
        }

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

def main():
    # Initialize the feature store
    fs = TaxiFeatureStore()
    
    # Load January 2025 data
    fs.load_data('data/yellow_tripdata_2025-01.parquet')
    
    # Test the demand features for a popular zone (e.g., Times Square area - zone 161)
    # Using a timestamp from the dataset
    test_timestamp = '2025-01-15 14:00:00'
    test_zone = 161
    
    print(f"\nTesting demand features for Zone {test_zone} at {test_timestamp}")
    features = fs.compute_demand_features(test_zone, test_timestamp, print_performance=True)
    
    print(f"Pickups in last hour: {features['pickups_last_hour']}")
    print(f"Pickups in last 3 hours: {features['pickups_last_3h']}")

    # Test next hour pickups prediction
    next_hour_pickups = fs.calculate_next_hour_pickups(test_zone, test_timestamp)
    print(f"\nPickups in next hour for Zone {test_zone} from {test_timestamp}:")
    print(f"Expected pickups in next hour: {next_hour_pickups}")
    

if __name__ == "__main__":
    main()