"""
Model serving module for taxi demand prediction.

This module provides functionality for training and serving machine learning
models to predict taxi pickup demand.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class WaitTimePredictor:
    """
    Machine learning model for predicting taxi pickup demand.
    
    This class handles training, evaluation, and prediction of taxi pickup
    demand using historical features.
    """
    
    def __init__(self, feature_store, config: Optional[Dict] = None):
        """
        Initialize the predictor.
        
        Args:
            feature_store: Instance of TaxiFeatureStore for feature computation
            config: Model configuration dictionary
        """
        self.feature_store = feature_store
        
        # Default configuration
        default_config = {
            'n_estimators': 100,
            'random_state': 42,
            'test_zones': [161, 162]  # Times Square, Midtown
        }
        self.config = {**default_config, **(config or {})}
        
        self.model = RandomForestRegressor(
            n_estimators=self.config['n_estimators'],
            random_state=self.config['random_state']
        )
        self._is_trained = False
        
    def prepare_training_data(
        self,
        zones: Optional[List[int]] = None,
        date: str = '2025-01-15'
    ) -> Tuple[List[List[Union[int, float]]], List[int]]:
        """
        Create dataset: past features → future pickups.
        
        Args:
            zones: List of zone IDs to include in training data
            date: Date string for training data generation
            
        Returns:
            Tuple of (features, targets) for training
            
        Raises:
            ValueError: If feature store is not properly initialized
        """
        if zones is None:
            zones = self.config['test_zones']
        
        if not hasattr(self.feature_store, 'df') or self.feature_store.df is None:
            raise ValueError("Feature store must have data loaded before preparing training data")
        
        X = []
        y = []
        
        try:
            # Convert date to datetime for iteration
            base_date = pd.to_datetime(date)
            
            # Iterate through each hour of the day (0-23)
            for hour in range(24):
                current_timestamp = base_date + timedelta(hours=hour)
                
                # For each zone, create training examples
                for zone_id in zones:
                    try:
                        # Get features at current time T
                        features = self.feature_store.compute_demand_features(zone_id, current_timestamp)
                        
                        # Get target: pickups in next hour (T to T+1)
                        next_hour_pickups = self.feature_store.calculate_next_hour_pickups(
                            zone_id, current_timestamp
                        )
                        
                        # Create feature vector: [zone_id, hour, pickups_last_hour, pickups_last_3h]
                        feature_vector = [
                            zone_id,
                            hour,
                            features['pickups_last_hour'],
                            features['pickups_last_3h']
                        ]
                        
                        X.append(feature_vector)
                        y.append(next_hour_pickups)
                        
                    except Exception as e:
                        logger.warning(f"Error processing zone {zone_id} at hour {hour}: {e}")
                        continue
            
            logger.info(f"Prepared {len(X)} training samples from {len(zones)} zones")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise

    def train(
        self,
        X: List[List[Union[int, float]]],
        y: List[int]
    ) -> Dict[str, float]:
        """
        Train the model with provided data.
        
        Args:
            X: Feature vectors
            y: Target values
            
        Returns:
            Dictionary with training statistics
            
        Raises:
            ValueError: If training data is invalid
        """
        if not X or not y:
            raise ValueError("Training data cannot be empty")
        
        if len(X) != len(y):
            raise ValueError("Features and targets must have the same length")
        
        try:
            # Convert to numpy arrays for scikit-learn
            X_array = np.array(X)
            y_array = np.array(y)
            
            # Train the model
            self.model.fit(X_array, y_array)
            self._is_trained = True
            
            # Calculate training metrics
            train_predictions = self.model.predict(X_array)
            
            stats = {
                'samples': len(X),
                'mae': mean_absolute_error(y_array, train_predictions),
                'mse': mean_squared_error(y_array, train_predictions),
                'rmse': np.sqrt(mean_squared_error(y_array, train_predictions))
            }
            
            logger.info(f"Model trained on {stats['samples']} samples")
            logger.info(f"Training MAE: {stats['mae']:.2f}")
            logger.info(f"Training RMSE: {stats['rmse']:.2f}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def predict(
        self,
        zone_id: int,
        hour: int,
        pickups_last_hour: int,
        pickups_last_3h: int
    ) -> float:
        """
        Make a single prediction.
        
        Args:
            zone_id: Zone ID for prediction
            hour: Hour of day (0-23)
            pickups_last_hour: Number of pickups in the last hour
            pickups_last_3h: Number of pickups in the last 3 hours
            
        Returns:
            Predicted number of pickups in the next hour
            
        Raises:
            ValueError: If model is not trained
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            feature_vector = np.array([[zone_id, hour, pickups_last_hour, pickups_last_3h]])
            prediction = self.model.predict(feature_vector)[0]
            return max(0, round(prediction))  # Ensure non-negative integer
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise

    def train_and_evaluate(
        self,
        train_date: str = '2025-01-15',
        test_date: str = '2025-01-16',
        zones: Optional[List[int]] = None
    ) -> Tuple[List[float], List[int], List[List[Union[int, float]]]]:
        """
        Train on one day, test on the next.
        
        Args:
            train_date: Date string for training data
            test_date: Date string for test data
            zones: List of zone IDs to use. If None, uses config default.
            
        Returns:
            Tuple of (predictions, actuals, test_features)
        """
        if zones is None:
            zones = self.config['test_zones']
        
        try:
            # Prepare training data
            logger.info(f"Preparing training data for {train_date}")
            X_train, y_train = self.prepare_training_data(zones=zones, date=train_date)
            
            # Train model
            logger.info("Training model...")
            train_stats = self.train(X_train, y_train)
            
            # Prepare test data
            logger.info(f"Preparing test data for {test_date}")
            X_test, y_test = self.prepare_training_data(zones=zones, date=test_date)
            
            # Make predictions
            logger.info("Making predictions...")
            predictions = []
            for features in X_test:
                pred = self.predict(
                    zone_id=features[0],
                    hour=features[1],
                    pickups_last_hour=features[2],
                    pickups_last_3h=features[3]
                )
                predictions.append(pred)
            
            # Calculate evaluation metrics
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            
            # Calculate MAPE with proper zero handling
            mape = self._calculate_mape(y_test, predictions)
            
            logger.info(f"Evaluation Results:")
            logger.info(f"  Mean Absolute Error (MAE): {mae:.2f}")
            logger.info(f"  Root Mean Square Error (RMSE): {rmse:.2f}")
            logger.info(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            logger.info(f"  Average actual pickups: {np.mean(y_test):.2f}")
            logger.info(f"  Average predicted pickups: {np.mean(predictions):.2f}")
            
            return predictions, y_test, X_test
            
        except Exception as e:
            logger.error(f"Error in train_and_evaluate: {e}")
            raise

    def _calculate_mape(self, actuals: List[int], predictions: List[float]) -> float:
        """
        Calculate Mean Absolute Percentage Error with proper zero handling.
        
        Args:
            actuals: Actual values
            predictions: Predicted values
            
        Returns:
            MAPE as a percentage
        """
        epsilon = 1e-8
        percentage_errors = []
        
        for pred, actual in zip(predictions, actuals):
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
        
        return np.mean(percentage_errors)

    def analyze_errors(
        self,
        predictions: List[float],
        actuals: List[int],
        X_test: List[List[Union[int, float]]]
    ) -> Dict[str, Union[int, float, Dict]]:
        """
        Analyze prediction errors to find patterns.
        
        Args:
            predictions: Model predictions
            actuals: Actual values
            X_test: Test features
            
        Returns:
            Dictionary with error analysis results
        """
        try:
            errors = [abs(p - a) for p, a in zip(predictions, actuals)]
            
            # Find worst prediction
            worst_idx = np.argmax(errors)
            worst_error = errors[worst_idx]
            
            # Calculate error statistics by zone and hour
            zone_errors = {}
            hour_errors = {}
            
            for i, (pred, actual, features) in enumerate(zip(predictions, actuals, X_test)):
                zone = int(features[0])
                hour = int(features[1])
                error = abs(pred - actual)
                
                if zone not in zone_errors:
                    zone_errors[zone] = []
                zone_errors[zone].append(error)
                
                if hour not in hour_errors:
                    hour_errors[hour] = []
                hour_errors[hour].append(error)
            
            # Calculate average errors
            avg_zone_errors = {zone: np.mean(errors) for zone, errors in zone_errors.items()}
            avg_hour_errors = {hour: np.mean(errors) for hour, errors in hour_errors.items()}
            
            results = {
                'worst_prediction': {
                    'index': worst_idx,
                    'zone': int(X_test[worst_idx][0]),
                    'hour': int(X_test[worst_idx][1]),
                    'predicted': predictions[worst_idx],
                    'actual': actuals[worst_idx],
                    'error': worst_error
                },
                'avg_zone_errors': avg_zone_errors,
                'avg_hour_errors': avg_hour_errors,
                'overall_mae': np.mean(errors),
                'overall_rmse': np.sqrt(np.mean([e**2 for e in errors]))
            }
            
            logger.info(f"Error Analysis:")
            logger.info(f"  Worst prediction: Zone {results['worst_prediction']['zone']}, "
                       f"Hour {results['worst_prediction']['hour']}")
            logger.info(f"  Predicted: {results['worst_prediction']['predicted']:.1f}, "
                       f"Actual: {results['worst_prediction']['actual']}")
            logger.info(f"  Error: {results['worst_prediction']['error']:.1f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing errors: {e}")
            raise

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores, or None if not trained
        """
        if not self._is_trained:
            logger.warning("Model is not trained, cannot get feature importance")
            return None
        
        feature_names = ['zone_id', 'hour', 'pickups_last_hour', 'pickups_last_3h']
        importance_scores = self.model.feature_importances_
        
        return dict(zip(feature_names, importance_scores))
