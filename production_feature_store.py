class ProductionFeatureStore:
    def __init__(self):
        # Offline store: For training (has all historical raw data)
        self.offline_store = PostgreSQL()  # All raw data
        
        # Online store: For serving (has only precomputed features)
        self.online_store = Redis()  # Just the features needed NOW
        
    def prepare_online_features(self, prediction_time):
        """Run this 5 minutes before each hourly prediction"""
        # Calculate features from offline store
        features = self.calculate_features_from_raw(prediction_time)
        
        # Cache in online store for fast serving
        self.online_store.set(f"features:{prediction_time}", features)