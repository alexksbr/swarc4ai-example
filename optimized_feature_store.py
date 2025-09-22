class OptimizedFeatureStore:
    def __init__(self):
        # Precompute hourly aggregates once
        self.hourly_stats = {}
        
    def precompute_hourly_stats(self, data):
        """Run this once on historical data"""
        for timestamp in data['timestamp'].unique():
            hourly_sum = data[data['timestamp'] == timestamp]['pickups'].sum()
            self.hourly_stats[timestamp] = hourly_sum
        print(f"Precomputed {len(self.hourly_stats)} hourly aggregates")
    
    def get_features(self, target_time):
        """Now this is FAST - just dictionary lookups!"""
        features = {
            'pickups_last_hour': self.hourly_stats.get(target_time - timedelta(hours=1), 0),
            'pickups_last_3h': sum([
                self.hourly_stats.get(target_time - timedelta(hours=i), 0) 
                for i in range(1, 4)
            ])
        }
        return features