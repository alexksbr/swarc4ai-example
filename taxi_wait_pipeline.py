# taxi_wait_time_pipeline.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

class TaxiWaitTimePipeline:
    def __init__(self):
        self.offline_store = {}
        self.online_store = {}
        self.model = None
        
    def generate_realistic_data(self):
        """Generate data with supply (drivers), demand (pickups), and wait times"""
        hours = pd.date_range('2024-01-01', '2024-03-31', freq='h')
        data = []
        
        for hour in hours:
            # Demand patterns
            base_demand = 100 + np.random.normal(0, 10)
            if hour.hour in [7, 8, 9, 17, 18, 19]:  # Rush hours
                pickups = int(base_demand * 2)
            elif hour.hour in [22, 23, 0, 1, 2]:  # Late night
                pickups = int(base_demand * 0.3)
            else:
                pickups = int(base_demand)
            
            # Supply patterns (drivers lag behind demand)
            base_supply = 30
            if hour.hour in [8, 9, 10, 18, 19, 20]:  # Drivers show up after rush starts
                drivers = int(base_supply * 1.5)
            elif hour.hour in [23, 0, 1, 2, 3]:  # Few drivers late night
                drivers = int(base_supply * 0.4)
            else:
                drivers = int(base_supply)
            
            # Calculate wait time using your formula
            workload = pickups / max(drivers, 1)
            base_wait = 2  # Minimum dispatch time
            wait_time = base_wait + (workload * 1.5)
            wait_time += np.random.normal(0, 1)  # Some variance
            wait_time = min(max(wait_time, 0), 30)  # Bound 0-30 min
            
            data.append({
                'timestamp': hour,
                'zone': 'TimesSquare',
                'pickups': pickups,
                'drivers': drivers,
                'wait_time': wait_time
            })
        
        df = pd.DataFrame(data)
        self.offline_store['raw_data'] = df
        print(f"Generated {len(df)} hours of wait time data")
        return df
    
    def _calculate_features(self, data, target_time):
        """Features for wait time prediction - includes supply AND demand"""
        
        # Time windows
        mask_1h = (data['timestamp'] > target_time - timedelta(hours=1)) & \
                  (data['timestamp'] <= target_time)
        mask_3h = (data['timestamp'] > target_time - timedelta(hours=3)) & \
                  (data['timestamp'] <= target_time)
        mask_yesterday = data['timestamp'] == (target_time - timedelta(days=1))
        
        features = {
            'hour': target_time.hour,
            'day_of_week': target_time.dayofweek,
            'is_weekend': target_time.dayofweek >= 5,
            
            # Demand features
            'recent_pickups': data[mask_1h]['pickups'].sum() if mask_1h.any() else 0,
            'pickups_3h': data[mask_3h]['pickups'].sum() if mask_3h.any() else 0,
            'pickups_yesterday': data[mask_yesterday]['pickups'].sum() if mask_yesterday.any() else 0,
            
            # Supply features
            'recent_drivers': data[mask_1h]['drivers'].mean() if mask_1h.any() else 30,
            'drivers_3h': data[mask_3h]['drivers'].mean() if mask_3h.any() else 30,
            
            # Key ratio feature - your workload insight
            'recent_workload': (data[mask_1h]['pickups'].sum() / 
                               max(data[mask_1h]['drivers'].mean(), 1)) if mask_1h.any() else 3
        }
        
        return features
    
    def train_model(self, start_date, end_date):
        """Train wait time prediction model"""
        print("Training wait time model...")
        
        raw_data = self.offline_store['raw_data']
        mask = (raw_data['timestamp'] >= start_date) & (raw_data['timestamp'] <= end_date)
        training_period = raw_data[mask].copy()
        
        features_list = []
        labels = []
        
        for idx, row in training_period.iterrows():
            target_time = row['timestamp']
            
            # Point-in-time correct features
            historical_data = raw_data[raw_data['timestamp'] < target_time]
            
            if len(historical_data) < 24:
                continue
                
            features = self._calculate_features(historical_data, target_time)
            features_list.append(features)
            labels.append(row['wait_time'])  # Now predicting wait time!
        
        X_train = pd.DataFrame(features_list)
        y_train = np.array(labels)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        print(f"Trained on {len(X_train)} examples")
        print(f"Average wait time in training: {np.mean(y_train):.1f} minutes")
        print(f"Feature importance:")
        for feat, imp in zip(X_train.columns, self.model.feature_importances_):
            print(f"  {feat}: {imp:.3f}")
        
        self.offline_store['model'] = self.model
        return self.model
    
    def prepare_serving_features(self, prediction_time):
        """Pre-compute features for serving"""
        raw_data = self.offline_store['raw_data']
        historical = raw_data[raw_data['timestamp'] < prediction_time]
        
        features = self._calculate_features(historical, prediction_time)
        
        cache_key = f"features:TimesSquare:{prediction_time}"
        self.online_store[cache_key] = features
        
        print(f"Cached features for {prediction_time}: workload={features['recent_workload']:.1f}")
        return features
    
    def serve_prediction(self, zone, current_time):
        """Predict wait time"""
        cache_key = f"features:{zone}:{current_time}"
        
        if cache_key not in self.online_store:
            print(f"Warning: No cached features for {current_time}")
            return None
        
        features = self.online_store[cache_key]
        feature_vector = pd.DataFrame([features])
        
        predicted_wait = self.model.predict(feature_vector)[0]
        
        return {
            'zone': zone,
            'time': current_time,
            'predicted_wait_minutes': round(predicted_wait, 1),
            'features_used': features
        }

# Test the refactored pipeline
if __name__ == "__main__":
    pipeline = TaxiWaitTimePipeline()
    
    # Generate realistic data
    pipeline.generate_realistic_data()
    
    # Train
    pipeline.train_model(
        start_date=pd.Timestamp('2024-02-01'),
        end_date=pd.Timestamp('2024-02-28')
    )
    
    # Prepare and serve
    test_time = pd.Timestamp('2024-03-01 17:00')  # Rush hour
    pipeline.prepare_serving_features(test_time)
    result = pipeline.serve_prediction('TimesSquare', test_time)
    print(f"\nPrediction: {result}")