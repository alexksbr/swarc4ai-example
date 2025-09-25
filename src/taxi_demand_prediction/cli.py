"""
Command-line interface for the taxi demand prediction system.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .utils import setup_logging, validate_zone_id
from .feature_store import TaxiFeatureStore
from .model_serving import WaitTimePredictor
from .monitoring import DemandMonitor
from .airport_predictor import AirportDemandPredictor
from .constants import DEFAULT_TEST_ZONES


def train_model(
    data_dir: str,
    train_date: str = "2025-01-15",
    test_date: str = "2025-01-16",
    zones: Optional[str] = None,
    output_dir: Optional[str] = None
) -> None:
    """
    Train and evaluate a model.
    
    Args:
        data_dir: Directory containing parquet files
        train_date: Date for training data
        test_date: Date for test data
        zones: Comma-separated zone IDs (default: 161,162)
        output_dir: Directory to save model results
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Parse zones
        if zones:
            zone_list = [int(z.strip()) for z in zones.split(",")]
            # Validate zone IDs
            invalid_zones = [z for z in zone_list if not validate_zone_id(z)]
            if invalid_zones:
                raise ValueError(f"Invalid zone IDs: {invalid_zones}")
        else:
            zone_list = DEFAULT_TEST_ZONES.copy()
        
        logger.info(f"Training model with zones: {zone_list}")
        
        # Initialize feature store and load data
        fs = TaxiFeatureStore({'data_dir': Path(data_dir)})
        fs.load_all_data()
        
        # Initialize predictor and monitor
        predictor = WaitTimePredictor(fs, {'test_zones': zone_list})
        monitor = DemandMonitor()
        
        # Train and evaluate
        logger.info(f"Training on {train_date}, testing on {test_date}")
        predictions, actuals, X_test = predictor.train_and_evaluate(
            train_date=train_date,
            test_date=test_date,
            zones=zone_list
        )
        
        # Log predictions to monitor
        for i, (pred, actual, features) in enumerate(zip(predictions, actuals, X_test)):
            zone = int(features[0])
            hour = int(features[1])
            monitor.log_prediction(zone=zone, hour=hour, predicted=pred, actual=actual)
        
        # Check for drift
        drift_detected = monitor.check_drift()
        if drift_detected:
            logger.warning("⚠️  Model drift detected!")
        else:
            logger.info("✅ No drift detected")
        
        # Analyze errors
        error_analysis = predictor.analyze_errors(predictions, actuals, X_test)
        
        # Get performance metrics
        performance = monitor.get_performance_metrics()
        logger.info(f"Performance: MAPE {performance.get('mape', 0):.2f}%")
        
        # Save results if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save performance summary
            import json
            summary = {
                'train_date': train_date,
                'test_date': test_date,
                'zones': zone_list,
                'performance': performance,
                'drift_detected': drift_detected,
                'error_analysis': error_analysis
            }
            
            with open(output_path / 'results.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Results saved to {output_path / 'results.json'}")
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        sys.exit(1)


def stress_test(
    data_dir: str,
    n_requests: int = 100,
    zones: Optional[str] = None
) -> None:
    """
    Run a stress test on the feature store.
    
    Args:
        data_dir: Directory containing parquet files
        n_requests: Number of requests to simulate
        zones: Comma-separated zone IDs
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Parse zones
        if zones:
            zone_list = [int(z.strip()) for z in zones.split(",")]
        else:
            zone_list = [161, 162, 237, 236]
        
        logger.info(f"Running stress test with {n_requests} requests on zones: {zone_list}")
        
        # Initialize feature store and load data
        fs = TaxiFeatureStore({'data_dir': Path(data_dir)})
        fs.load_all_data()
        
        # Run stress test
        stats = fs.stress_test(n_requests=n_requests, zones=zone_list)
        
        logger.info("Stress test completed successfully")
        
    except Exception as e:
        logger.error(f"Error running stress test: {e}")
        sys.exit(1)


def compare_demand(
    data_dir: str,
    zone_id: int = 161,
    current_time: Optional[str] = None
) -> None:
    """
    Compare airport demand with city demand for a specific zone.
    
    Args:
        data_dir: Directory containing parquet files
        zone_id: Zone ID to compare (default: 161 - Times Square)
        current_time: Current time in format 'YYYY-MM-DD HH:MM:SS' (default: current time)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Parse current time
        if current_time:
            parsed_time = datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
        else:
            # Use a sample time from our data
            parsed_time = datetime.strptime('2025-01-15 14:00:00', '%Y-%m-%d %H:%M:%S')
        
        logger.info(f"Comparing demand for zone {zone_id} at {parsed_time}")
        
        # Initialize feature store and load data
        fs = TaxiFeatureStore({'data_dir': Path(data_dir)})
        fs.load_all_data()
        
        # Train a model for predictions
        logger.info("Training model for demand predictions...")
        predictor = WaitTimePredictor(fs, {'test_zones': [zone_id]})
        X_train, y_train = predictor.prepare_training_data(zones=[zone_id], date='2025-01-15')
        predictor.train(X_train, y_train)
        logger.info("Model training completed")
        
        # Initialize airport predictor with trained model
        airport_predictor = AirportDemandPredictor(fs, predictor)
        
        # Compare locations
        comparison = airport_predictor.compare_locations(zone_id, parsed_time)
        
        # Display results
        logger.info("=" * 50)
        logger.info("DEMAND COMPARISON RESULTS")
        logger.info("=" * 50)
        logger.info(f"Zone ID: {comparison['city_zone']}")
        logger.info(f"City Demand (next hour): {comparison['city_demand']} pickups")
        logger.info(f"Airport Demand (next hour): {comparison['airport_demand']} pickups")
        logger.info(f"Recommendation: {comparison['recommendation']}")
        
        if comparison['recommendation'] == 'AIRPORT':
            logger.info("💡 Driver should go to the airport for better opportunities")
        else:
            logger.info("💡 Driver should stay in the current zone")
        
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error running demand comparison: {e}")
        sys.exit(1)


def test_revenue(
    data_dir: str,
    zone_id: int = 161
) -> None:
    """
    Test revenue recommendations at different times of day.
    
    Args:
        data_dir: Directory containing parquet files
        zone_id: Zone ID to test (default: 161 - Times Square)
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Testing revenue recommendations for zone {zone_id}")
        
        # Initialize feature store and load data
        fs = TaxiFeatureStore({'data_dir': Path(data_dir)})
        fs.load_all_data()
        
        # Train a model for predictions
        logger.info("Training model for demand predictions...")
        predictor = WaitTimePredictor(fs, {'test_zones': [zone_id]})
        X_train, y_train = predictor.prepare_training_data(zones=[zone_id], date='2025-01-15')
        predictor.train(X_train, y_train)
        logger.info("Model training completed")
        
        # Initialize airport predictor with trained model
        airport_predictor = AirportDemandPredictor(fs, predictor)
        
        # Run revenue tests
        results = airport_predictor.test_revenue_recommendations(zone_id)
        
        # Display results
        logger.info("=" * 60)
        logger.info("REVENUE RECOMMENDATIONS BY TIME OF DAY")
        logger.info("=" * 60)
        logger.info(f"Zone ID: {zone_id}")
        logger.info("")
        
        for result in results:
            revenue = result['revenue']
            time_display = result['time_display']
            recommendation = revenue['revenue_recommendation']
            difference = revenue['revenue_difference']
            
            logger.info(f"{time_display}:")
            logger.info(f"  City: ${revenue['city_revenue_per_hour']:.2f}/hr ({revenue['city_trips_per_hour']:.1f} trips)")
            logger.info(f"  Airport: ${revenue['airport_revenue_per_hour']:.2f}/hr ({revenue['airport_trips_per_hour']:.1f} trips)")
            logger.info(f"  → {recommendation} (${difference:.2f}/hr difference)")
            
            if recommendation == 'AIRPORT':
                logger.info("    💰 Airport is more profitable")
            else:
                logger.info("    🏙️ Stay in city for better earnings")
            logger.info("")
        
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error running revenue test: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Taxi Demand Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level"
    )
    
    parser.add_argument(
        "--log-file",
        help="Path to log file (default: console only)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train and evaluate model")
    train_parser.add_argument("data_dir", help="Directory containing parquet files")
    train_parser.add_argument("--train-date", default="2025-01-15", help="Training date")
    train_parser.add_argument("--test-date", default="2025-01-16", help="Test date")
    train_parser.add_argument("--zones", help="Comma-separated zone IDs (e.g., 161,162)")
    train_parser.add_argument("--output-dir", help="Directory to save results")
    
    # Stress test command
    stress_parser = subparsers.add_parser("stress", help="Run stress test")
    stress_parser.add_argument("data_dir", help="Directory containing parquet files")
    stress_parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    stress_parser.add_argument("--zones", help="Comma-separated zone IDs")
    
    # Compare demand command
    compare_parser = subparsers.add_parser("compare", help="Compare city vs airport demand")
    compare_parser.add_argument("data_dir", help="Directory containing parquet files")
    compare_parser.add_argument("--zone-id", type=int, default=161, help="Zone ID to compare (default: 161 - Times Square)")
    compare_parser.add_argument("--time", help="Current time in format 'YYYY-MM-DD HH:MM:SS' (default: sample time)")
    
    # Test revenue command
    revenue_parser = subparsers.add_parser("test-revenue", help="Test revenue recommendations throughout the day")
    revenue_parser.add_argument("data_dir", help="Directory containing parquet files")
    revenue_parser.add_argument("--zone-id", type=int, default=161, help="Zone ID to test (default: 161 - Times Square)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level, log_file=args.log_file)
    
    if args.command == "train":
        train_model(
            data_dir=args.data_dir,
            train_date=args.train_date,
            test_date=args.test_date,
            zones=args.zones,
            output_dir=args.output_dir
        )
    elif args.command == "stress":
        stress_test(
            data_dir=args.data_dir,
            n_requests=args.requests,
            zones=args.zones
        )
    elif args.command == "compare":
        compare_demand(
            data_dir=args.data_dir,
            zone_id=args.zone_id,
            current_time=args.time
        )
    elif args.command == "test-revenue":
        test_revenue(
            data_dir=args.data_dir,
            zone_id=args.zone_id
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
