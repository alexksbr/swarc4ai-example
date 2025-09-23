# taxi_demand_pipeline.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle

class TaxiDemandPipeline:
    def __init__(self):
        self.offline_store = {}  # Simulating PostgreSQL
        self.online_store = {}   # Simulating Redis
        self.model = None
        
    def _calculate_features(self, data, target_time):
        """SHARED feature logic - this is your key insight!
        Used by BOTH training and serving to ensure consistency"""
        
        # Critical: use < target_time to avoid leakage
        mask_1h = (data['timestamp'] > target_time - timedelta(hours=1)) & \
                  (data['timestamp'] <= target_time)
        mask_3h = (data['timestamp'] > target_time - timedelta(hours=3)) & \
                  (data['timestamp'] <= target_time)
        mask_yesterday = data['timestamp'] == (target_time - timedelta(days=1))
        
        features = {
            'hour': target_time.hour,  # Cyclical pattern feature
            'day_of_week': target_time.dayofweek,
            'pickups_last_1h': data[mask_1h]['pickups'].sum() if mask_1h.any() else 0,
            'pickups_last_3h': data[mask_3h]['pickups'].sum() if mask_3h.any() else 0,
            'pickups_yesterday': data[mask_yesterday]['pickups'].sum() if mask_yesterday.any() else 0,
        }
        return features
    
    def load_historical_data(self, csv_path=None):
        """Load data into offline store"""
        # Simulate with random data for now
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='h')
        data = pd.DataFrame({
            'timestamp': dates,
            'zone': 'TimesSquare',
            'pickups': np.random.poisson(100 + 50*np.sin(np.arange(len(dates))/24), len(dates))
        })
        self.offline_store['raw_data'] = data
        print(f"Loaded {len(data)} hours of historical data")
        return data
    
    def train_model(self, start_date, end_date):
        """Create training data with point-in-time correct features"""
        print("Creating training dataset...")
        
        # Get raw data from offline store
        raw_data = self.offline_store['raw_data']
        
        # Filter to training period
        mask = (raw_data['timestamp'] >= start_date) & (raw_data['timestamp'] <= end_date)
        training_period = raw_data[mask].copy()
        
        # Generate features for each hour (this is where point-in-time matters!)
        features_list = []
        labels = []
        
        for idx, row in training_period.iterrows():
            target_time = row['timestamp']
            
            # Only use data BEFORE this point for features
            historical_data = raw_data[raw_data['timestamp'] < target_time]
            
            if len(historical_data) < 24:  # Need at least 1 day of history
                continue
                
            # Use shared feature calculation
            features = self._calculate_features(historical_data, target_time)
            features_list.append(features)
            labels.append(row['pickups'])
        
        # Convert to DataFrame for training
        X_train = pd.DataFrame(features_list)
        y_train = np.array(labels)
        
        # Simple model - in production you'd use XGBoost/LightGBM
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        print(f"Trained on {len(X_train)} examples")
        print(f"Feature importance: {dict(zip(X_train.columns, self.model.feature_importances_))}")
        
        # Store model in offline store (in production: model registry)
        self.offline_store['model'] = self.model
        
        return self.model
    
    def prepare_serving_features(self, prediction_time):
        """Prepare features for next hour's prediction
        This runs ~5 min before each hour in production"""
        
        print(f"Preparing features for {prediction_time}")
        
        # Get recent data from offline store
        raw_data = self.offline_store['raw_data']
        historical = raw_data[raw_data['timestamp'] < prediction_time]
        
        # Use THE SAME feature calculation as training!
        features = self._calculate_features(historical, prediction_time)
        
        # Cache in online store with TTL (in production: Redis with expiry)
        cache_key = f"features:TimesSquare:{prediction_time}"
        self.online_store[cache_key] = features
        
        print(f"Cached features: {features}")
        return features
    
    def serve_prediction(self, zone, current_time):
        """Make real-time prediction - must be FAST (<100ms)"""
        
        # Fetch from online store (this would be Redis GET - very fast)
        cache_key = f"features:{zone}:{current_time}"
        
        if cache_key not in self.online_store:
            # Fallback: use last available features (accept staleness)
            print(f"Warning: No pre-computed features for {current_time}, using fallback")
            # In production: fetch most recent cached features
            return None
        
        features = self.online_store[cache_key]
        
        # Transform to model input
        feature_vector = pd.DataFrame([features])
        
        # Predict (in production: this model would be loaded in memory)
        prediction = self.model.predict(feature_vector)[0]
        
        return {
            'zone': zone,
            'time': current_time,
            'predicted_pickups': int(prediction),
            'features_used': features
        }

# Let's test it!
if __name__ == "__main__":
    pipeline = TaxiDemandPipeline()
    
    # Step 1: Load historical data
    pipeline.load_historical_data()
    
    # Step 2: Train model (ensuring point-in-time correctness)
    pipeline.train_model(
        start_date=pd.Timestamp('2024-02-01'),
        end_date=pd.Timestamp('2024-02-28')
    )
    
    # Step 3: Prepare features for serving (runs before prediction time)
    prediction_time = pd.Timestamp('2024-03-01 15:00')
    pipeline.prepare_serving_features(prediction_time)
    
    # Step 4: Serve real-time prediction
    result = pipeline.serve_prediction('TimesSquare', prediction_time)
    print(f"\nPrediction result: {result}")