import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class TaxiFeatureStore:
    def __init__(self):
        self.df = None
        
    def load_data(self, parquet_path):
        """Load the parquet file"""
        self.df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(self.df)} trips")
        
    def compute_demand_features(self, zone_id, timestamp):
        """
        Calculate demand features for a zone at a specific time
        """
        # Filter to the zone
        zone_trips = self.df[self.df['PULocationID'] == zone_id]
        
        if not pd.api.types.is_datetime64_any_dtype(zone_trips['tpep_pickup_datetime']):
            zone_trips = zone_trips.copy()
            zone_trips['tpep_pickup_datetime'] = pd.to_datetime(zone_trips['tpep_pickup_datetime'])

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
        
        return {
            'pickups_last_hour': pickups_last_hour,
            'pickups_last_3h': pickups_last_3h
        }

    def calculate_wait_times(self, zone_id, date):
        """
        Calculate actual wait times from the data
        """
        # Filter to specific zone and date
        zone_trips = self.df[self.df['PULocationID'] == zone_id].copy()
        zone_trips['pickup_date'] = zone_trips['tpep_pickup_datetime'].dt.date
        zone_trips = zone_trips[zone_trips['pickup_date'] == pd.to_datetime(date).date()]
        
        if len(zone_trips) < 2:
            return []
        
        # Sort by pickup time
        zone_trips = zone_trips.sort_values('tpep_pickup_datetime')
        
        wait_times = zone_trips['tpep_pickup_datetime'].diff().dropna().dt.total_seconds().div(60).tolist()
        
        return wait_times

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
    features = fs.compute_demand_features(test_zone, test_timestamp)
    
    print(f"Pickups in last hour: {features['pickups_last_hour']}")
    print(f"Pickups in last 3 hours: {features['pickups_last_3h']}")

    # Test wait times
    test_date = '2025-01-15'
    wait_times = fs.calculate_wait_times(test_zone, test_date)
    print(f"\nWait times for Zone {test_zone} on {test_date}:")
    print(f"Average wait: {np.mean(wait_times):.1f} minutes")
    print(f"Max wait: {np.max(wait_times):.1f} minutes")

if __name__ == "__main__":
    main()