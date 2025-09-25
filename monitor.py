import numpy as np
from datetime import datetime, timedelta

class DemandMonitor:
    def __init__(self, alert_threshold=30):  # 30% MAPE threshold
        self.predictions = []
        self.actuals = []
        self.alert_threshold = alert_threshold
        
    def log_prediction(self, zone, hour, predicted, actual=None):
        """Log each prediction and optionally its actual outcome"""
        self.predictions.append({
            'timestamp': datetime.now(),
            'zone': zone,
            'hour': hour,
            'predicted': predicted,
            'actual': actual
        })
        
    def check_drift(self, window_size=20):
        """
        YOUR TASK: Check if recent predictions are drifting
        Calculate rolling MAPE for last 'window_size' predictions
        Alert if above threshold
        """
        # Get recent predictions with known actuals
        recent = [p for p in self.predictions[-window_size:] if p['actual'] is not None]
        
        if len(recent) < 10:  # Need minimum samples
            return False

        # Calculate MAPE (Mean Absolute Percentage Error)
        mape_values = []
        for prediction in recent:
            actual = prediction['actual']
            predicted = prediction['predicted']
            
            # Avoid division by zero
            if actual != 0:
                mape_values.append(abs((actual - predicted) / actual) * 100)
        
        # Calculate mean MAPE
        mape = np.mean(mape_values) if mape_values else 0
        
        return mape > self.alert_threshold