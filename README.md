# Taxi Demand Prediction System

A comprehensive system for predicting taxi pickup demand using historical data, feature engineering, machine learning models, and real-time monitoring capabilities.

## Features

- **Feature Engineering**: Extract demand features from historical taxi trip data with intelligent caching
- **Machine Learning**: Train and deploy Random Forest models for pickup demand prediction
- **Real-time Monitoring**: Track model performance, detect drift, and generate alerts
- **High Performance**: Optimized feature computation with caching and stress testing capabilities
- **Comprehensive Testing**: Full test suite with pytest
- **CLI Interface**: Command-line tools for training and testing
- **Proper Logging**: Configurable logging throughout the application

## Project Structure

```
swarc4ai-example/
├── src/taxi_demand_prediction/          # Main package
│   ├── __init__.py                     # Package initialization
│   ├── feature_store.py                # Feature engineering and caching
│   ├── model_serving.py                # ML model training and serving
│   ├── monitoring.py                   # Performance monitoring and alerting
│   ├── utils.py                        # Utility functions
│   └── cli.py                          # Command-line interface
├── tests/                              # Test suite
│   ├── test_feature_store.py          # Feature store tests
│   ├── test_monitoring.py             # Monitoring tests
│   └── __init__.py
├── config/                             # Configuration management
│   └── config.py                       # Configuration classes
├── data/                               # Data directory
│   ├── yellow_tripdata_2025-01.parquet
│   ├── yellow_tripdata_2025-02.parquet
│   └── yellow_tripdata_2025-03.parquet
├── pyproject.toml                      # Project configuration
├── README.md                           # This file
└── main.py                            # Legacy entry point
```

## Installation

### Requirements

- Python 3.12+
- Dependencies listed in `pyproject.toml`

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd swarc4ai-example
   ```

2. **Install dependencies:**
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -e .
   
   # For development with additional tools
   pip install -e ".[dev]"
   ```

3. **Verify installation:**
   ```bash
   taxi-predict --help
   ```

## Usage

### Command Line Interface

The system provides a CLI for common operations:

#### Train and Evaluate a Model

```bash
# Basic training
taxi-predict train data/

# With custom parameters
taxi-predict train data/ \
    --train-date 2025-01-15 \
    --test-date 2025-01-16 \
    --zones 161,162,237 \
    --output-dir results/ \
    --log-level DEBUG
```

#### Run Stress Tests

```bash
# Basic stress test
taxi-predict stress data/

# Custom stress test
taxi-predict stress data/ \
    --requests 500 \
    --zones 161,162,237,236
```

### Python API

You can also use the system programmatically:

```python
from taxi_demand_prediction import TaxiFeatureStore, WaitTimePredictor, DemandMonitor
from taxi_demand_prediction.utils import setup_logging

# Setup logging
setup_logging(level="INFO", log_file="taxi_prediction.log")

# Initialize components
feature_store = TaxiFeatureStore()
feature_store.load_all_data("data/")

predictor = WaitTimePredictor(feature_store)
monitor = DemandMonitor()

# Train and evaluate
predictions, actuals, features = predictor.train_and_evaluate(
    train_date="2025-01-15",
    test_date="2025-01-16"
)

# Monitor predictions
for i, (pred, actual, feat) in enumerate(zip(predictions, actuals, features)):
    monitor.log_prediction(
        zone=int(feat[0]),
        hour=int(feat[1]),
        predicted=pred,
        actual=actual
    )

# Check for drift
if monitor.check_drift():
    print("⚠️  Model drift detected!")

# Get performance metrics
metrics = monitor.get_performance_metrics()
print(f"MAPE: {metrics['mape']:.2f}%")
```

## Core Components

### TaxiFeatureStore

Handles feature engineering with intelligent caching:

- **Data Loading**: Load multiple parquet files with validation
- **Feature Computation**: Calculate demand features (pickups in last 1h, 3h)
- **Caching**: TTL-based caching for performance optimization
- **Stress Testing**: Performance testing capabilities

Key methods:
- `load_data(parquet_path)`: Load a single parquet file
- `load_all_data(data_dir)`: Load all configured parquet files
- `compute_demand_features(zone_id, timestamp)`: Compute features with caching
- `calculate_next_hour_pickups(zone_id, timestamp)`: Get target values
- `stress_test(n_requests)`: Performance testing

### WaitTimePredictor

Machine learning model for demand prediction:

- **Training**: Train Random Forest models on historical data
- **Evaluation**: Comprehensive model evaluation with multiple metrics
- **Prediction**: Make real-time predictions
- **Error Analysis**: Detailed error analysis by zone and time

Key methods:
- `train_and_evaluate(train_date, test_date)`: Full training pipeline
- `predict(zone_id, hour, features)`: Make single prediction
- `analyze_errors(predictions, actuals)`: Error analysis
- `get_feature_importance()`: Feature importance scores

### DemandMonitor

Real-time monitoring and alerting:

- **Prediction Logging**: Track all predictions and outcomes
- **Drift Detection**: Detect model performance degradation
- **Alerting**: Generate alerts based on configurable thresholds
- **Performance Metrics**: Calculate MAPE, MAE, RMSE over time windows
- **Data Management**: Automatic cleanup of old data

Key methods:
- `log_prediction(zone, hour, predicted, actual)`: Log predictions
- `check_drift(window_size)`: Check for model drift
- `get_performance_metrics(hours_back)`: Calculate metrics
- `get_alerts(hours_back, level)`: Retrieve alerts
- `clear_old_data(days_to_keep)`: Data cleanup

## Configuration

The system uses configuration classes for flexible setup:

```python
from config.config import AppConfig, DataConfig, ModelConfig, MonitoringConfig

# Custom configuration
config = AppConfig(
    data=DataConfig(
        data_dir=Path("custom_data/"),
        cache_ttl_hours=2.0
    ),
    model=ModelConfig(
        n_estimators=200,
        test_zones=[161, 162, 237]
    ),
    monitoring=MonitoringConfig(
        alert_threshold_mape=25.0,
        drift_window_size=30
    )
)
```

## Testing

The project includes comprehensive tests using pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_feature_store.py

# Run with verbose output
pytest -v
```

### Test Coverage

- **Feature Store**: Data loading, feature computation, caching, error handling
- **Monitoring**: Prediction logging, drift detection, alerting, metrics calculation
- **Integration**: End-to-end workflows

## Performance Considerations

### Caching Strategy

The feature store implements intelligent caching:

- **TTL-based expiration**: Configurable cache lifetime
- **Memory management**: Automatic cleanup of old entries
- **Cache statistics**: Monitor cache hit rates

### Optimization Tips

1. **Batch Processing**: Load all data files at startup
2. **Cache Tuning**: Adjust TTL based on your use case
3. **Memory Management**: Use `clear_old_data()` for long-running processes
4. **Zone Selection**: Focus on high-traffic zones for better performance

## Monitoring and Alerting

### Drift Detection

The system automatically detects model drift using MAPE thresholds:

```python
# Configure drift detection
monitor = DemandMonitor({
    'alert_threshold_mape': 30.0,  # 30% MAPE threshold
    'drift_window_size': 20,       # Last 20 predictions
    'min_samples_for_drift': 10    # Minimum samples needed
})

# Check for drift
if monitor.check_drift():
    print("Model needs retraining!")
```

### Alert Levels

- **INFO**: Informational messages
- **WARNING**: Performance degradation detected
- **CRITICAL**: Severe issues requiring immediate attention

### Metrics

The system tracks comprehensive metrics:

- **MAPE**: Mean Absolute Percentage Error
- **MAE**: Mean Absolute Error  
- **RMSE**: Root Mean Square Error
- **Zone-specific metrics**: Performance by location
- **Time-based analysis**: Performance trends over time

## Development

### Code Quality

The project includes tools for maintaining code quality:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure parquet files exist in the data directory
2. **Memory Issues**: Use `clear_old_data()` for long-running processes
3. **Import Errors**: Verify the package is properly installed with `pip install -e .`
4. **Performance Issues**: Check cache hit rates and adjust TTL settings

### Logging

Enable debug logging for troubleshooting:

```python
from taxi_demand_prediction.utils import setup_logging

setup_logging(level="DEBUG", log_file="debug.log")
```

### Support

For issues and questions:

1. Check the test suite for usage examples
2. Review the API documentation in docstrings
3. Enable debug logging for detailed information
4. Create an issue in the repository

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with scikit-learn for machine learning
- Uses pandas and numpy for data processing
- Parquet format for efficient data storage
- pytest for comprehensive testing
