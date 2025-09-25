"""
Command-line interface for the taxi demand prediction system.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .utils import setup_logging
from .feature_store import TaxiFeatureStore
from .model_serving import WaitTimePredictor
from .monitoring import DemandMonitor


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
        else:
            zone_list = [161, 162]
        
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
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
