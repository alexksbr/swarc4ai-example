"""
Legacy main entry point for backwards compatibility.

This file maintains backwards compatibility with the old structure.
For new development, use the CLI: `taxi-predict --help`
"""

import sys
import warnings
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from taxi_demand_prediction.utils import setup_logging
from taxi_demand_prediction.feature_store import TaxiFeatureStore
from taxi_demand_prediction.model_serving import WaitTimePredictor
from taxi_demand_prediction.monitoring import DemandMonitor


def main():
    """
    Legacy main function - runs a basic demo of the system.
    
    For full functionality, use the CLI:
        taxi-predict train data/
        taxi-predict stress data/
    """
    warnings.warn(
        "This legacy entry point is deprecated. "
        "Use 'taxi-predict --help' for the full CLI interface.",
        DeprecationWarning,
        stacklevel=2
    )
    
    print("🚕 Taxi Demand Prediction System")
    print("=" * 50)
    
    # Setup logging
    setup_logging(level="INFO")
    
    try:
        # Initialize components
        print("Initializing feature store...")
        fs = TaxiFeatureStore()
        
        # Check if data files exist
        data_dir = Path("data")
        if not data_dir.exists():
            print("❌ Data directory not found. Please ensure 'data/' directory exists with parquet files.")
            return
        
        # Load data
        print("Loading data...")
        try:
            fs.load_all_data()
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("Please ensure parquet files exist in the data/ directory.")
            return
        
        print(f"✅ Loaded data successfully")
        
        # Initialize predictor and monitor
        predictor = WaitTimePredictor(fs)
        monitor = DemandMonitor()
        
        # Run a quick demo
        print("\nRunning demo prediction...")
        
        # Get features for Times Square at noon
        features = fs.compute_demand_features(161, '2025-01-15 12:00:00')
        print(f"Features for Times Square (zone 161) at noon: {features}")
        
        # Calculate next hour pickups
        next_hour = fs.calculate_next_hour_pickups(161, '2025-01-15 12:00:00')
        print(f"Actual pickups in next hour: {next_hour}")
        
        # Quick training demo (if data is available)
        print("\nTesting model training...")
        try:
            predictions, actuals, X_test = predictor.train_and_evaluate(
                train_date='2025-01-15',
                test_date='2025-01-16'
            )
            
            print(f"✅ Training completed successfully!")
            print(f"Generated {len(predictions)} predictions")
            
            # Log some predictions to monitor
            for i in range(min(10, len(predictions))):
                monitor.log_prediction(
                    zone=int(X_test[i][0]),
                    hour=int(X_test[i][1]),
                    predicted=predictions[i],
                    actual=actuals[i]
                )
            
            # Check performance
            metrics = monitor.get_performance_metrics()
            if 'mape' in metrics:
                print(f"Model MAPE: {metrics['mape']:.2f}%")
            
            # Check for drift
            drift = monitor.check_drift()
            print(f"Drift detected: {'Yes' if drift else 'No'}")
            
        except Exception as e:
            print(f"⚠️  Training demo failed: {e}")
            print("This is normal if you don't have the expected date ranges in your data.")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"\nFor full functionality, use the CLI:")
        print(f"  taxi-predict train data/")
        print(f"  taxi-predict stress data/")
        print(f"  taxi-predict --help")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
