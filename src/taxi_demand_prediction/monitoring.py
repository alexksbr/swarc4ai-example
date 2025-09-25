"""
Monitoring module for taxi demand prediction system.

This module provides functionality for monitoring model performance,
detecting drift, and alerting on anomalies.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .constants import (
    DEFAULT_ALERT_THRESHOLD_MAPE, DEFAULT_DRIFT_WINDOW_SIZE,
    DEFAULT_MIN_SAMPLES_FOR_DRIFT, DEFAULT_PERFORMANCE_WINDOW_HOURS,
    DEFAULT_ALERT_COOLDOWN_MINUTES, DEFAULT_DATA_RETENTION_DAYS
)
from .utils import calculate_mape, validate_zone_id, validate_hour

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PredictionRecord:
    """Record for a single prediction."""
    timestamp: datetime
    zone: int
    hour: int
    predicted: float
    actual: Optional[int] = None
    features: Optional[Dict] = None
    model_version: Optional[str] = None


@dataclass
class Alert:
    """Alert record."""
    timestamp: datetime
    level: AlertLevel
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    details: Optional[Dict] = None


class DemandMonitor:
    """
    Monitor for taxi demand prediction system.
    
    This class tracks predictions, calculates performance metrics,
    detects model drift, and generates alerts.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the monitor.
        
        Args:
            config: Monitoring configuration dictionary
        """
        # Default configuration
        default_config = {
            'alert_threshold_mape': DEFAULT_ALERT_THRESHOLD_MAPE,
            'drift_window_size': DEFAULT_DRIFT_WINDOW_SIZE,
            'min_samples_for_drift': DEFAULT_MIN_SAMPLES_FOR_DRIFT,
            'performance_window_hours': DEFAULT_PERFORMANCE_WINDOW_HOURS,
            'alert_cooldown_minutes': DEFAULT_ALERT_COOLDOWN_MINUTES,
        }
        self.config = {**default_config, **(config or {})}
        
        self.predictions: List[PredictionRecord] = []
        self.alerts: List[Alert] = []
        self._last_alert_times: Dict[str, datetime] = {}
        
    def log_prediction(
        self,
        zone: int,
        hour: int,
        predicted: float,
        actual: Optional[int] = None,
        features: Optional[Dict] = None,
        model_version: Optional[str] = None
    ) -> None:
        """
        Log a prediction for monitoring.
        
        Args:
            zone: Zone ID
            hour: Hour of day (0-23)
            predicted: Predicted value
            actual: Actual value (if available)
            features: Features used for prediction
            model_version: Version of the model used
            
        Raises:
            ValueError: If zone or hour is invalid
        """
        if not validate_zone_id(zone):
            raise ValueError(f"Invalid zone ID: {zone}")
        
        if not validate_hour(hour):
            raise ValueError(f"Invalid hour: {hour}")
        
        try:
            record = PredictionRecord(
                timestamp=datetime.now(),
                zone=zone,
                hour=hour,
                predicted=predicted,
                actual=actual,
                features=features,
                model_version=model_version
            )
            
            self.predictions.append(record)
            logger.debug(f"Logged prediction for zone {zone}, hour {hour}: {predicted}")
            
            # Check for immediate alerts if actual is available
            if actual is not None:
                self._check_prediction_quality(record)
                
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")

    def update_actual(self, prediction_index: int, actual: int) -> None:
        """
        Update the actual value for a logged prediction.
        
        Args:
            prediction_index: Index of the prediction to update
            actual: Actual observed value
            
        Raises:
            IndexError: If prediction_index is invalid
        """
        try:
            if 0 <= prediction_index < len(self.predictions):
                self.predictions[prediction_index].actual = actual
                logger.debug(f"Updated actual value for prediction {prediction_index}: {actual}")
                
                # Check for alerts with the new actual value
                self._check_prediction_quality(self.predictions[prediction_index])
            else:
                raise IndexError(f"Invalid prediction index: {prediction_index}")
                
        except Exception as e:
            logger.error(f"Error updating actual value: {e}")
            raise

    def check_drift(self, window_size: Optional[int] = None) -> bool:
        """
        Check if recent predictions are drifting.
        
        Args:
            window_size: Number of recent predictions to analyze
            
        Returns:
            True if drift is detected, False otherwise
        """
        if window_size is None:
            window_size = self.config['drift_window_size']
        
        try:
            # Get recent predictions with known actuals
            recent = [p for p in self.predictions[-window_size:] if p.actual is not None]
            
            if len(recent) < self.config['min_samples_for_drift']:
                logger.debug(f"Not enough samples for drift detection: {len(recent)}")
                return False

            # Calculate MAPE for recent predictions
            mape = calculate_mape([p.actual for p in recent], [p.predicted for p in recent])
            threshold = self.config['alert_threshold_mape']
            
            drift_detected = mape > threshold
            
            if drift_detected:
                self._create_alert(
                    level=AlertLevel.WARNING,
                    message=f"Model drift detected: MAPE {mape:.2f}% exceeds threshold {threshold}%",
                    metric_name="mape",
                    metric_value=mape,
                    threshold=threshold,
                    details={"window_size": len(recent), "samples": len(recent)}
                )
                logger.warning(f"Drift detected: MAPE {mape:.2f}% > {threshold}%")
            else:
                logger.debug(f"No drift detected: MAPE {mape:.2f}% <= {threshold}%")
            
            return drift_detected
            
        except Exception as e:
            logger.error(f"Error checking drift: {e}")
            return False

    def get_performance_metrics(
        self,
        hours_back: Optional[int] = None,
        zone_id: Optional[int] = None
    ) -> Dict[str, Union[float, int, Dict]]:
        """
        Calculate performance metrics for recent predictions.
        
        Args:
            hours_back: Number of hours to look back (default from config)
            zone_id: Specific zone to analyze (None for all zones)
            
        Returns:
            Dictionary with performance metrics
        """
        if hours_back is None:
            hours_back = self.config['performance_window_hours']
        
        try:
            # Filter predictions by time and zone
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            filtered_predictions = [
                p for p in self.predictions
                if p.timestamp >= cutoff_time and p.actual is not None
                and (zone_id is None or p.zone == zone_id)
            ]
            
            if not filtered_predictions:
                return {"error": "No predictions with actuals in the specified time window"}
            
            actuals = [p.actual for p in filtered_predictions]
            predictions = [p.predicted for p in filtered_predictions]
            
            # Calculate metrics
            mae = np.mean([abs(a - p) for a, p in zip(actuals, predictions)])
            mse = np.mean([(a - p)**2 for a, p in zip(actuals, predictions)])
            rmse = np.sqrt(mse)
            mape = calculate_mape(actuals, predictions)
            
            # Zone-specific metrics
            zone_metrics = {}
            zones = set(p.zone for p in filtered_predictions)
            for zone in zones:
                zone_preds = [p for p in filtered_predictions if p.zone == zone]
                zone_actuals = [p.actual for p in zone_preds]
                zone_predictions = [p.predicted for p in zone_preds]
                
                zone_metrics[zone] = {
                    'count': len(zone_preds),
                    'mae': np.mean([abs(a - p) for a, p in zip(zone_actuals, zone_predictions)]),
                    'mape': calculate_mape(zone_actuals, zone_predictions)
                }
            
            metrics = {
                'sample_count': len(filtered_predictions),
                'time_window_hours': hours_back,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'avg_actual': np.mean(actuals),
                'avg_predicted': np.mean(predictions),
                'zone_metrics': zone_metrics,
                'zones_analyzed': list(zones)
            }
            
            logger.info(f"Performance metrics calculated for {len(filtered_predictions)} predictions")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {"error": str(e)}

    def get_alerts(
        self,
        hours_back: Optional[int] = None,
        level: Optional[AlertLevel] = None
    ) -> List[Alert]:
        """
        Get recent alerts.
        
        Args:
            hours_back: Number of hours to look back (None for all alerts)
            level: Filter by alert level (None for all levels)
            
        Returns:
            List of alerts matching the criteria
        """
        try:
            filtered_alerts = self.alerts
            
            if hours_back is not None:
                cutoff_time = datetime.now() - timedelta(hours=hours_back)
                filtered_alerts = [a for a in filtered_alerts if a.timestamp >= cutoff_time]
            
            if level is not None:
                filtered_alerts = [a for a in filtered_alerts if a.level == level]
            
            return sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []

    def clear_old_data(self, days_to_keep: int = DEFAULT_DATA_RETENTION_DAYS) -> Dict[str, int]:
        """
        Clear old predictions and alerts to manage memory.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Dictionary with counts of removed items
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)
            
            # Count items before removal
            old_pred_count = len(self.predictions)
            old_alert_count = len(self.alerts)
            
            # Remove old data
            self.predictions = [p for p in self.predictions if p.timestamp >= cutoff_time]
            self.alerts = [a for a in self.alerts if a.timestamp >= cutoff_time]
            
            removed_counts = {
                'predictions_removed': old_pred_count - len(self.predictions),
                'alerts_removed': old_alert_count - len(self.alerts),
                'predictions_remaining': len(self.predictions),
                'alerts_remaining': len(self.alerts)
            }
            
            logger.info(f"Cleaned old data: removed {removed_counts['predictions_removed']} "
                       f"predictions and {removed_counts['alerts_removed']} alerts")
            
            return removed_counts
            
        except Exception as e:
            logger.error(f"Error clearing old data: {e}")
            return {"error": str(e)}


    def _check_prediction_quality(self, record: PredictionRecord) -> None:
        """
        Check the quality of a single prediction and create alerts if needed.
        
        Args:
            record: Prediction record to check
        """
        if record.actual is None:
            return
        
        try:
            # Calculate error for this prediction
            error = abs(record.actual - record.predicted)
            percentage_error = calculate_mape([record.actual], [record.predicted])
            
            # Check if error is unusually high
            if percentage_error > self.config['alert_threshold_mape'] * 2:  # 2x threshold for single prediction
                self._create_alert(
                    level=AlertLevel.WARNING,
                    message=f"High prediction error for zone {record.zone}, hour {record.hour}: "
                           f"{percentage_error:.1f}% error",
                    metric_name="single_prediction_mape",
                    metric_value=percentage_error,
                    threshold=self.config['alert_threshold_mape'] * 2,
                    details={
                        "zone": record.zone,
                        "hour": record.hour,
                        "predicted": record.predicted,
                        "actual": record.actual,
                        "absolute_error": error
                    }
                )
                
        except Exception as e:
            logger.error(f"Error checking prediction quality: {e}")

    def _create_alert(
        self,
        level: AlertLevel,
        message: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
        details: Optional[Dict] = None
    ) -> None:
        """
        Create and store an alert.
        
        Args:
            level: Alert severity level
            message: Alert message
            metric_name: Name of the metric that triggered the alert
            metric_value: Current value of the metric
            threshold: Threshold that was exceeded
            details: Additional details about the alert
        """
        try:
            # Check cooldown to avoid spam
            alert_key = f"{metric_name}_{level.value}"
            now = datetime.now()
            
            if alert_key in self._last_alert_times:
                time_since_last = now - self._last_alert_times[alert_key]
                cooldown = timedelta(minutes=self.config['alert_cooldown_minutes'])
                
                if time_since_last < cooldown:
                    logger.debug(f"Alert {alert_key} still in cooldown, skipping")
                    return
            
            # Create the alert
            alert = Alert(
                timestamp=now,
                level=level,
                message=message,
                metric_name=metric_name,
                metric_value=metric_value,
                threshold=threshold,
                details=details
            )
            
            self.alerts.append(alert)
            self._last_alert_times[alert_key] = now
            
            # Log the alert
            log_level = {
                AlertLevel.INFO: logging.INFO,
                AlertLevel.WARNING: logging.WARNING,
                AlertLevel.CRITICAL: logging.CRITICAL
            }[level]
            
            logger.log(log_level, f"ALERT [{level.value.upper()}]: {message}")
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")

    def get_summary(self) -> Dict[str, Union[int, float, Dict]]:
        """
        Get a summary of monitoring statistics.
        
        Returns:
            Dictionary with monitoring summary
        """
        try:
            now = datetime.now()
            last_24h = now - timedelta(hours=24)
            
            # Count predictions and alerts
            total_predictions = len(self.predictions)
            predictions_24h = len([p for p in self.predictions if p.timestamp >= last_24h])
            predictions_with_actuals = len([p for p in self.predictions if p.actual is not None])
            
            total_alerts = len(self.alerts)
            alerts_24h = len([a for a in self.alerts if a.timestamp >= last_24h])
            
            # Alert breakdown by level
            alert_levels = {}
            for level in AlertLevel:
                alert_levels[level.value] = len([a for a in self.alerts if a.level == level])
            
            # Get recent performance if available
            recent_performance = self.get_performance_metrics(hours_back=24)
            
            summary = {
                'total_predictions': total_predictions,
                'predictions_last_24h': predictions_24h,
                'predictions_with_actuals': predictions_with_actuals,
                'total_alerts': total_alerts,
                'alerts_last_24h': alerts_24h,
                'alert_breakdown': alert_levels,
                'cache_size': len(self.predictions),
                'monitoring_since': min([p.timestamp for p in self.predictions]) if self.predictions else None,
                'last_prediction': max([p.timestamp for p in self.predictions]) if self.predictions else None,
                'recent_performance': recent_performance if 'error' not in recent_performance else None
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {"error": str(e)}
