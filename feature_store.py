# feature_store.py
import pandas as pd
from datetime import timedelta
import sqlite3

class SimpleFeatureStore:
    def __init__(self):
        self.conn = sqlite3.connect('features.db')
        
    def create_features_for_training(self, data, target_times):
        """Create point-in-time correct features for training"""
        features = []
        
        for target_time in target_times:
            # Calculate pickups in the last hour (exclusive of target_time, inclusive of 1 hour before)
            mask_last_hour = (data['timestamp'] > (target_time - timedelta(hours=1))) & (data['timestamp'] <= target_time)
            pickups_last_hour = data.loc[mask_last_hour, 'pickups'].sum()

            # Calculate pickups in the last 3 hours (exclusive of target_time, inclusive of 3 hours before)
            mask_last_3h = (data['timestamp'] > (target_time - timedelta(hours=3))) & (data['timestamp'] <= target_time)
            pickups_last_3h = data.loc[mask_last_3h, 'pickups'].sum()

            # Calculate pickups at the same hour yesterday
            time_yesterday = target_time - timedelta(days=1)
            mask_yesterday = data['timestamp'] == time_yesterday
            pickups_yesterday = data.loc[mask_yesterday, 'pickups'].sum()
            
            row = {
                'timestamp': target_time,
                'pickups_last_hour': pickups_last_hour,  # You calculate this
                'pickups_last_3h': pickups_last_3h,     # And this
                'pickups_yesterday': pickups_yesterday    # And this
            }
            features.append(row)
            
        return pd.DataFrame(features)