"""
Test script for the taxi pickup prediction model with monitoring
"""
from feature_store import TaxiFeatureStore
from model_serving import WaitTimePredictor
from monitor import DemandMonitor


def main():
    """
    Test the train_and_evaluate function with real taxi data and monitoring
    """
    print("=== Testing Pickup Prediction Model with Monitoring ===")
    
    # Initialize feature store and load data
    print("Loading taxi data...")
    fs = TaxiFeatureStore()
    
    # Load all three months of data
    fs.load_data('data/yellow_tripdata_2025-01.parquet')
    fs.load_data('data/yellow_tripdata_2025-02.parquet')
    fs.load_data('data/yellow_tripdata_2025-03.parquet')
    
    # Initialize predictor and monitor
    predictor = WaitTimePredictor(fs)
    monitor = DemandMonitor(alert_threshold=30)  # 30% MAPE threshold
    
    # Test train_and_evaluate function
    print("\nTraining on 2025-01-15 and testing on 2025-01-16...")
    predictions, actuals, X_test = predictor.train_and_evaluate(
        train_date='2025-01-15', 
        test_date='2025-01-16'
    )
    
    print(f"\nGenerated {len(predictions)} test predictions")
    print(f"Sample predictions: {predictions[:5]}")
    print(f"Sample actuals: {actuals[:5]}")
    
    # Show some prediction vs actual comparisons
    print(f"\nFirst 5 prediction comparisons:")
    for i in range(min(5, len(predictions))):
        print(f"  Predicted: {predictions[i]:.1f}, Actual: {actuals[i]}, Error: {abs(predictions[i] - actuals[i]):.1f}")
    
    # Analyze prediction errors
    print(f"\nAnalyzing prediction errors...")
    predictor.analyze_errors(predictions, actuals, X_test)
    
    # Test monitoring functionality
    print(f"\n=== Testing Monitoring System ===")
    
    # Log all predictions to the monitor
    print("Logging predictions to monitor...")
    for i, (pred, actual, features) in enumerate(zip(predictions, actuals, X_test)):
        zone = int(features[0])
        hour = int(features[1])
        monitor.log_prediction(zone=zone, hour=hour, predicted=pred, actual=actual)
    
    print(f"Logged {len(monitor.predictions)} predictions to monitor")
    
    # Check for drift
    print("\nChecking for model drift...")
    drift_detected = monitor.check_drift(window_size=20)
    
    if drift_detected:
        print("⚠️  ALERT: Model drift detected! MAPE above threshold.")
    else:
        print("✅ No drift detected. Model performance is stable.")
    
    # Show monitoring statistics
    recent_predictions = monitor.predictions[-20:]  # Last 20 predictions
    recent_with_actuals = [p for p in recent_predictions if p['actual'] is not None]
    
    if recent_with_actuals:
        recent_mape_values = []
        for p in recent_with_actuals:
            if p['actual'] != 0:
                mape = abs((p['actual'] - p['predicted']) / p['actual']) * 100
                recent_mape_values.append(mape)
        
        if recent_mape_values:
            avg_recent_mape = sum(recent_mape_values) / len(recent_mape_values)
            print(f"Average MAPE for last {len(recent_with_actuals)} predictions: {avg_recent_mape:.2f}%")
            print(f"Monitor threshold: {monitor.alert_threshold}%")
    
    # Simulate real-time monitoring scenario
    print(f"\n=== Simulating Real-time Monitoring ===")
    print("Making new predictions and monitoring them...")
    
    # Make a few new predictions for different zones/times
    test_scenarios = [
        {'zone': 161, 'time': '2025-01-17 08:00:00', 'description': 'Times Square - Morning'},
        {'zone': 162, 'time': '2025-01-17 12:00:00', 'description': 'Midtown - Lunch'},
        {'zone': 161, 'time': '2025-01-17 20:00:00', 'description': 'Times Square - Evening'},
    ]
    
    for scenario in test_scenarios:
        # Get features for prediction (with performance metrics during inference)
        features = fs.compute_demand_features(scenario['zone'], scenario['time'], print_performance=True)
        hour = int(scenario['time'].split(' ')[1].split(':')[0])
        
        # Create feature vector for prediction
        feature_vector = [[
            scenario['zone'],
            hour,
            features['pickups_last_hour'],
            features['pickups_last_3h']
        ]]
        
        # Make prediction
        predicted = predictor.model.predict(feature_vector)[0]
        predicted = max(0, round(predicted))
        
        # Get actual for comparison (if available in data)
        actual = fs.calculate_next_hour_pickups(scenario['zone'], scenario['time'])
        
        # Log to monitor
        monitor.log_prediction(
            zone=scenario['zone'], 
            hour=hour, 
            predicted=predicted, 
            actual=actual
        )
        
        print(f"{scenario['description']}: Predicted {predicted}, Actual {actual}")
    
    # Final drift check
    final_drift = monitor.check_drift(window_size=25)
    print(f"\nFinal drift check: {'DRIFT DETECTED' if final_drift else 'NO DRIFT'}")
    print(f"Total predictions monitored: {len(monitor.predictions)}")


if __name__ == "__main__":
    main()
