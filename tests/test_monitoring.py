"""
Unit tests for the DemandMonitor class.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from src.taxi_demand_prediction.monitoring import DemandMonitor, AlertLevel, PredictionRecord, Alert


@pytest.fixture
def monitor():
    """Create a DemandMonitor instance for testing."""
    return DemandMonitor()


@pytest.fixture
def monitor_with_config():
    """Create a DemandMonitor with custom config."""
    config = {
        'alert_threshold_mape': 25.0,
        'drift_window_size': 10,
        'min_samples_for_drift': 5,
        'alert_cooldown_minutes': 30
    }
    return DemandMonitor(config)


class TestDemandMonitor:
    """Test cases for DemandMonitor."""
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = DemandMonitor()
        
        assert monitor.predictions == []
        assert monitor.alerts == []
        assert monitor._last_alert_times == {}
        assert 'alert_threshold_mape' in monitor.config
        assert 'drift_window_size' in monitor.config
        assert 'min_samples_for_drift' in monitor.config
    
    def test_initialization_with_config(self, monitor_with_config):
        """Test monitor initialization with custom config."""
        assert monitor_with_config.config['alert_threshold_mape'] == 25.0
        assert monitor_with_config.config['drift_window_size'] == 10
        assert monitor_with_config.config['min_samples_for_drift'] == 5
    
    def test_log_prediction_without_actual(self, monitor):
        """Test logging prediction without actual value."""
        monitor.log_prediction(zone=161, hour=12, predicted=5.0)
        
        assert len(monitor.predictions) == 1
        record = monitor.predictions[0]
        
        assert record.zone == 161
        assert record.hour == 12
        assert record.predicted == 5.0
        assert record.actual is None
        assert isinstance(record.timestamp, datetime)
    
    def test_log_prediction_with_actual(self, monitor):
        """Test logging prediction with actual value."""
        monitor.log_prediction(zone=161, hour=12, predicted=5.0, actual=6)
        
        assert len(monitor.predictions) == 1
        record = monitor.predictions[0]
        
        assert record.zone == 161
        assert record.hour == 12
        assert record.predicted == 5.0
        assert record.actual == 6
    
    def test_log_prediction_with_features(self, monitor):
        """Test logging prediction with features."""
        features = {'pickups_last_hour': 3, 'pickups_last_3h': 8}
        monitor.log_prediction(
            zone=161, 
            hour=12, 
            predicted=5.0, 
            actual=6,
            features=features,
            model_version="v1.0"
        )
        
        record = monitor.predictions[0]
        assert record.features == features
        assert record.model_version == "v1.0"
    
    def test_update_actual_valid_index(self, monitor):
        """Test updating actual value with valid index."""
        monitor.log_prediction(zone=161, hour=12, predicted=5.0)
        monitor.update_actual(0, 6)
        
        assert monitor.predictions[0].actual == 6
    
    def test_update_actual_invalid_index(self, monitor):
        """Test updating actual value with invalid index."""
        monitor.log_prediction(zone=161, hour=12, predicted=5.0)
        
        with pytest.raises(IndexError):
            monitor.update_actual(1, 6)  # Index out of range
    
    def test_check_drift_insufficient_samples(self, monitor):
        """Test drift detection with insufficient samples."""
        # Add a few predictions but not enough for drift detection
        for i in range(3):
            monitor.log_prediction(zone=161, hour=12, predicted=5.0, actual=5)
        
        drift = monitor.check_drift()
        assert drift == False
    
    def test_check_drift_no_drift(self, monitor):
        """Test drift detection when no drift is present."""
        # Add predictions with good accuracy
        for i in range(15):
            monitor.log_prediction(zone=161, hour=12, predicted=5.0, actual=5)
        
        drift = monitor.check_drift()
        assert drift == False
    
    def test_check_drift_detected(self, monitor):
        """Test drift detection when drift is present."""
        # Add predictions with poor accuracy (high MAPE)
        for i in range(15):
            monitor.log_prediction(zone=161, hour=12, predicted=5.0, actual=20)  # High error
        
        drift = monitor.check_drift()
        assert drift == True
        
        # Should have created an alert
        assert len(monitor.alerts) > 0
        alert = monitor.alerts[-1]
        assert alert.level == AlertLevel.WARNING
        assert "drift detected" in alert.message.lower()
    
    def test_get_performance_metrics_empty(self, monitor):
        """Test performance metrics with no data."""
        metrics = monitor.get_performance_metrics()
        assert "error" in metrics
    
    def test_get_performance_metrics_success(self, monitor):
        """Test performance metrics calculation."""
        # Add some predictions with actuals
        predictions_data = [
            (161, 12, 5.0, 5),
            (161, 13, 6.0, 7),
            (162, 12, 3.0, 2),
            (162, 13, 4.0, 5)
        ]
        
        for zone, hour, pred, actual in predictions_data:
            monitor.log_prediction(zone=zone, hour=hour, predicted=pred, actual=actual)
        
        metrics = monitor.get_performance_metrics()
        
        assert 'sample_count' in metrics
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert 'avg_actual' in metrics
        assert 'avg_predicted' in metrics
        assert 'zone_metrics' in metrics
        
        assert metrics['sample_count'] == 4
        assert metrics['mae'] >= 0
        assert metrics['mape'] >= 0
        assert len(metrics['zone_metrics']) == 2  # Two zones
    
    def test_get_performance_metrics_zone_filter(self, monitor):
        """Test performance metrics with zone filtering."""
        # Add predictions for different zones
        monitor.log_prediction(zone=161, hour=12, predicted=5.0, actual=5)
        monitor.log_prediction(zone=162, hour=12, predicted=6.0, actual=6)
        
        # Get metrics for zone 161 only
        metrics = monitor.get_performance_metrics(zone_id=161)
        
        assert metrics['sample_count'] == 1
        assert 161 in metrics['zones_analyzed']
        assert 162 not in metrics['zones_analyzed']
    
    def test_get_alerts_all(self, monitor):
        """Test getting all alerts."""
        # Manually add some alerts
        alert1 = Alert(
            timestamp=datetime.now(),
            level=AlertLevel.WARNING,
            message="Test alert 1",
            metric_name="test",
            metric_value=50.0,
            threshold=30.0
        )
        alert2 = Alert(
            timestamp=datetime.now(),
            level=AlertLevel.CRITICAL,
            message="Test alert 2",
            metric_name="test",
            metric_value=80.0,
            threshold=30.0
        )
        
        monitor.alerts.extend([alert1, alert2])
        
        alerts = monitor.get_alerts()
        assert len(alerts) == 2
    
    def test_get_alerts_filtered_by_level(self, monitor):
        """Test getting alerts filtered by level."""
        # Add alerts with different levels
        alert1 = Alert(
            timestamp=datetime.now(),
            level=AlertLevel.WARNING,
            message="Warning alert",
            metric_name="test",
            metric_value=50.0,
            threshold=30.0
        )
        alert2 = Alert(
            timestamp=datetime.now(),
            level=AlertLevel.CRITICAL,
            message="Critical alert",
            metric_name="test",
            metric_value=80.0,
            threshold=30.0
        )
        
        monitor.alerts.extend([alert1, alert2])
        
        # Get only warning alerts
        warning_alerts = monitor.get_alerts(level=AlertLevel.WARNING)
        assert len(warning_alerts) == 1
        assert warning_alerts[0].level == AlertLevel.WARNING
    
    def test_get_alerts_filtered_by_time(self, monitor):
        """Test getting alerts filtered by time."""
        # Add an old alert
        old_alert = Alert(
            timestamp=datetime.now() - timedelta(hours=25),
            level=AlertLevel.WARNING,
            message="Old alert",
            metric_name="test",
            metric_value=50.0,
            threshold=30.0
        )
        
        # Add a recent alert
        recent_alert = Alert(
            timestamp=datetime.now(),
            level=AlertLevel.WARNING,
            message="Recent alert",
            metric_name="test",
            metric_value=50.0,
            threshold=30.0
        )
        
        monitor.alerts.extend([old_alert, recent_alert])
        
        # Get alerts from last 24 hours
        recent_alerts = monitor.get_alerts(hours_back=24)
        assert len(recent_alerts) == 1
        assert recent_alerts[0].message == "Recent alert"
    
    def test_clear_old_data(self, monitor):
        """Test clearing old data."""
        # Add old predictions
        old_time = datetime.now() - timedelta(days=8)
        with patch('src.taxi_demand_prediction.monitoring.datetime') as mock_datetime:
            mock_datetime.now.return_value = old_time
            monitor.log_prediction(zone=161, hour=12, predicted=5.0, actual=5)
        
        # Add recent predictions
        monitor.log_prediction(zone=161, hour=13, predicted=6.0, actual=6)
        
        # Add old alert
        old_alert = Alert(
            timestamp=old_time,
            level=AlertLevel.WARNING,
            message="Old alert",
            metric_name="test",
            metric_value=50.0,
            threshold=30.0
        )
        monitor.alerts.append(old_alert)
        
        # Clear old data (keep 7 days)
        result = monitor.clear_old_data(days_to_keep=7)
        
        assert result['predictions_removed'] == 1
        assert result['alerts_removed'] == 1
        assert result['predictions_remaining'] == 1
        assert result['alerts_remaining'] == 0
    
    def test_get_summary(self, monitor):
        """Test getting monitoring summary."""
        # Add some test data
        monitor.log_prediction(zone=161, hour=12, predicted=5.0, actual=5)
        monitor.log_prediction(zone=162, hour=13, predicted=6.0, actual=6)
        
        summary = monitor.get_summary()
        
        assert 'total_predictions' in summary
        assert 'predictions_last_24h' in summary
        assert 'predictions_with_actuals' in summary
        assert 'total_alerts' in summary
        assert 'alerts_last_24h' in summary
        assert 'alert_breakdown' in summary
        assert 'monitoring_since' in summary
        assert 'last_prediction' in summary
        
        assert summary['total_predictions'] == 2
        assert summary['predictions_with_actuals'] == 2
