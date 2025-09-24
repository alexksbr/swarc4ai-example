from taxi_wait_pipeline import TaxiWaitTimePipeline
from datetime import timedelta
import pandas as pd


class RealTimeMonitoredPipeline(TaxiWaitTimePipeline):
    def __init__(self):
        super().__init__()
        self.active_predictions = {}
        self.monitoring_alerts = []
        self.alert_threshold = 1.5  # Your choice!
        
    def check_active_predictions(self, current_time):
        """Run this every minute to check predictions"""
        alerts_triggered = []
        
        for pred_time, pred_info in list(self.active_predictions.items()):
            elapsed = (current_time - pred_time).total_seconds() / 60
            
            # Your 1.5x rule
            if elapsed > pred_info['predicted'] * self.alert_threshold and not pred_info['alerted']:
                alert = {
                    'prediction_time': pred_time,
                    'predicted_wait': pred_info['predicted'],
                    'elapsed_so_far': elapsed,
                    'alert_time': current_time,
                    'severity': 'HIGH' if elapsed > pred_info['predicted'] * 2 else 'MEDIUM'
                }
                self.monitoring_alerts.append(alert)
                pred_info['alerted'] = True
                alerts_triggered.append(alert)
                
            # Clean up old predictions (after 30 min, assume completed)
            if elapsed > 30:
                del self.active_predictions[pred_time]
        
        return alerts_triggered
    
    def simulate_concert_scenario(self):
        """Let's see your monitoring in action!"""
        print("\n=== SIMULATING CONCERT NIGHT ===\n")
        
        # Normal predictions first
        normal_time = pd.Timestamp('2024-03-01 18:00')
        self.prepare_serving_features(normal_time)
        result = self.serve_prediction_with_monitoring('TimesSquare', normal_time)
        print(f"18:00 - Normal evening prediction: {result['predicted_wait_minutes']} min")
        
        # Concert ends at 21:00 - sudden spike!
        concert_time = pd.Timestamp('2024-03-01 21:00')
        
        # Inject concert data (600 pickups, still only 30 drivers)
        concert_features = {
            'hour': 21,
            'day_of_week': 4,
            'is_weekend': False,
            'recent_pickups': 600,  # SPIKE!
            'pickups_3h': 250,
            'pickups_yesterday': 100,
            'recent_drivers': 30,    # No change!
            'drivers_3h': 30,
            'recent_workload': 20.0  # 600/30 = 20!
        }
        
        cache_key = f"features:TimesSquare:{concert_time}"
        self.online_store[cache_key] = concert_features
        
        result = self.serve_prediction_with_monitoring('TimesSquare', concert_time)
        print(f"21:00 - Concert ends, model predicts: {result['predicted_wait_minutes']} min")
        print(f"       (Workload is {concert_features['recent_workload']:.1f} - way above normal!)")
        
        # Now simulate time passing...
        print("\n--- Monitoring Active Predictions ---")
        for minutes_later in [5, 10, 12, 15, 20]:
            check_time = concert_time + timedelta(minutes=minutes_later)
            alerts = self.check_active_predictions(check_time)
            
            if alerts:
                for alert in alerts:
                    print(f"⚠️  ALERT at 21:{minutes_later:02d}!")
                    print(f"   Predicted: {alert['predicted_wait']:.1f} min")
                    print(f"   Already waiting: {alert['elapsed_so_far']:.1f} min")
            else:
                print(f"21:{minutes_later:02d} - Monitoring... no alerts yet")
    
    def calculate_dynamic_scale_factor(self):
        """Your idea: scale based on recent errors"""
        recent_errors = []

        for pred_time, pred_info in self.active_predictions.items():
            if 'actual_wait' in pred_info:  # We know the outcome
                error_ratio = pred_info['actual_wait'] / pred_info['predicted']
                recent_errors.append(error_ratio)
    
        if len(recent_errors) >= 5:  # Need enough data
            return np.mean(recent_errors[-10:])  # Last 10 errors
        return 1.0  # No scaling yet