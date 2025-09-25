import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

class WaitTimePredictor:
    def __init__(self, feature_store):
        self.feature_store = feature_store
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def prepare_training_data(self, zones=[161, 162], date='2025-01-15'):
        """
        Create dataset: past features → future pickups
        """
        X = []
        y = []
        
        # Convert date to datetime for iteration
        base_date = pd.to_datetime(date)
        
        # Iterate through each hour of the day (0-23)
        for hour in range(24):
            current_timestamp = base_date + timedelta(hours=hour)
            
            # For each zone, create training examples
            for zone_id in zones:
                # Get features at current time T
                features = self.feature_store.compute_demand_features(zone_id, current_timestamp)
                
                # Get target: pickups in next hour (T to T+1)
                next_hour_pickups = self.feature_store.calculate_next_hour_pickups(zone_id, current_timestamp)
                
                # Create feature vector: [zone_id, hour, pickups_last_hour, pickups_last_3h]
                feature_vector = [
                    zone_id,
                    hour,
                    features['pickups_last_hour'],
                    features['pickups_last_3h']
                ]
                
                X.append(feature_vector)
                y.append(next_hour_pickups)

        return X, y

    def train_and_evaluate(self, train_date='2025-01-15', test_date='2025-01-16'):
        """
        Train on one day, test on the next
        """
        # Prepare training data
        X_train, y_train = self.prepare_training_data(zones=[161, 162], date=train_date)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Prepare test data
        X_test, y_test = self.prepare_training_data(zones=[161, 162], date=test_date)
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Calculate Mean Absolute Percentage Error (MAPE)
        # Handle division by zero by adding small epsilon
        epsilon = 1e-8
        percentage_errors = []
        
        for pred, actual in zip(predictions, y_test):
            if actual == 0 and pred == 0:
                # Both are zero, perfect prediction
                percentage_errors.append(0)
            elif actual == 0:
                # Actual is zero but prediction is not, use absolute error
                percentage_errors.append(abs(pred))
            else:
                # Standard MAPE calculation
                percentage_error = abs((actual - pred) / (actual + epsilon)) * 100
                percentage_errors.append(percentage_error)
        
        mape = np.mean(percentage_errors)
        
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"Average actual pickups: {np.mean(y_test):.2f}")
        print(f"Average predicted pickups: {np.mean(predictions):.2f}")
        
        return predictions, y_test, X_test

    def analyze_errors(self, predictions, actuals, X_test):
        """
        Find patterns in the errors
        Which zones/hours have the worst predictions?
        """
        errors = [abs(p - a) for p, a in zip(predictions, actuals)]
        
        # Print the worst predictions
        worst_idx = np.argmax(errors)
        print(f"Worst prediction:")
        print(f"  Features: zone={X_test[worst_idx][0]}, hour={X_test[worst_idx][1]}")
        print(f"  Predicted: {predictions[worst_idx]:.1f}, Actual: {actuals[worst_idx]}")

