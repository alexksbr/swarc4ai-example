import random
from datetime import datetime, timedelta

class AirportDemandPredictor:
    def __init__(self, taxi_feature_store, wait_time_predictor=None):
        self.taxi_store = taxi_feature_store
        self.wait_time_predictor = wait_time_predictor
        
    def generate_flight_arrivals(self, current_time):
        """Simple flight generator - busier during day"""
        hour = current_time.hour
        
        # More flights during day (6AM-10PM)
        if 6 <= hour <= 22:
            num_flights = random.randint(3, 8)
        else:
            num_flights = random.randint(0, 2)
            
        flights = []
        for _ in range(num_flights):
            flights.append({
                'arrival_time': current_time + timedelta(minutes=random.randint(30, 90)),
                'passengers': random.randint(100, 300),
                'international': random.random() > 0.7,  # 30% international
            })
        return flights
    
    def estimate_airport_demand(self, current_time):
        """Estimate taxis needed at airport in next ~hour"""
        flights = self.generate_flight_arrivals(current_time)
        
        taxi_demand = 0
        for flight in flights:
            # Assume 30% of passengers take taxis
            taxi_passengers = flight['passengers'] * 0.3
            
            # International flights = more taxi usage
            if flight['international']:
                taxi_passengers *= 1.2
                
            taxi_demand += taxi_passengers
            
        return int(taxi_demand)
    
    def _get_city_demand_prediction(self, zone_id, current_time):
        """Get city demand prediction using the trained model or fallback to historical data."""
        if self.wait_time_predictor and self.wait_time_predictor._is_trained:
            # Use model inference
            try:
                # Get current features for prediction
                features = self.taxi_store.compute_demand_features(zone_id, current_time)
                hour = current_time.hour
                
                # Make prediction using the trained model
                prediction = self.wait_time_predictor.predict(
                    zone_id=zone_id,
                    hour=hour,
                    pickups_last_hour=features['pickups_last_hour'],
                    pickups_last_3h=features['pickups_last_3h']
                )
                return prediction
            except Exception as e:
                # Fallback to historical data if model prediction fails
                print(f"Warning: Model prediction failed ({e}), falling back to historical data")
                return self.taxi_store.calculate_next_hour_pickups(zone_id, current_time)
        else:
            # Fallback to historical data if no trained model available
            print("Warning: No trained model available, using historical data")
            return self.taxi_store.calculate_next_hour_pickups(zone_id, current_time)
    
    def compare_locations(self, zone_id, current_time):
        """Should driver stay in zone or go to airport?"""
        # City demand from your predictor (using model inference)
        city_demand = self._get_city_demand_prediction(zone_id, current_time)
        
        # Airport demand
        airport_demand = self.estimate_airport_demand(current_time)
        
        # Simple decision - where's more demand?
        return {
            'city_zone': zone_id,
            'city_demand': city_demand,
            'airport_demand': airport_demand,
            'recommendation': 'AIRPORT' if airport_demand > city_demand else 'STAY'
        }
    
    def calculate_expected_revenue(self, zone_id, current_time):
        """Calculate expected revenue per hour for city vs airport"""
        # Parse current_time if it's a string
        if isinstance(current_time, str):
            current_time = datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
        
        # Get demand estimates (using model inference)
        city_demand = self._get_city_demand_prediction(zone_id, current_time)
        airport_demand = self.estimate_airport_demand(current_time)
        
        # Revenue assumptions (based on NYC taxi data)
        city_avg_fare = 15.50  # Average city fare
        city_avg_trip_time = 12  # minutes per trip
        max_city_trips_per_hour = 60 / city_avg_trip_time  # Maximum trips limited by time
        
        airport_avg_fare = 45.00  # Higher fare to/from airport
        airport_avg_trip_time = 35  # Longer trips to airport
        max_airport_trips_per_hour = 60 / airport_avg_trip_time  # Maximum trips limited by time
        
        # Calculate actual trips per hour based on demand and time constraints
        # Higher demand means higher probability of getting rides quickly
        # Use a more realistic model: trips = min(max_trips, demand_factor * max_trips)
        city_demand_factor = min(1.0, city_demand / 500.0)  # Scale demand to 0-1, normalize around 500 pickups
        airport_demand_factor = min(1.0, airport_demand / 200.0)  # Scale demand to 0-1, normalize around 200 pickups
        
        city_trips_per_hour = min(max_city_trips_per_hour, city_demand_factor * max_city_trips_per_hour)
        airport_trips_per_hour = min(max_airport_trips_per_hour, airport_demand_factor * max_airport_trips_per_hour)
        
        # Calculate hourly revenue
        city_revenue_per_hour = city_trips_per_hour * city_avg_fare
        airport_revenue_per_hour = airport_trips_per_hour * airport_avg_fare
        
        return {
            'city_zone': zone_id,
            'city_demand': city_demand,
            'city_trips_per_hour': city_trips_per_hour,
            'city_avg_fare': city_avg_fare,
            'city_revenue_per_hour': city_revenue_per_hour,
            'airport_demand': airport_demand,
            'airport_trips_per_hour': airport_trips_per_hour,
            'airport_avg_fare': airport_avg_fare,
            'airport_revenue_per_hour': airport_revenue_per_hour,
            'revenue_recommendation': 'AIRPORT' if airport_revenue_per_hour > city_revenue_per_hour else 'STAY',
            'revenue_difference': abs(airport_revenue_per_hour - city_revenue_per_hour)
        }
    
    def test_revenue_recommendations(self, zone_id=161):
        """Test revenue model at different times of day"""
        test_times = [
            "2025-01-15 06:00:00",  # Early morning
            "2025-01-15 09:00:00",  # Morning rush
            "2025-01-15 14:00:00",  # Afternoon
            "2025-01-15 18:00:00",  # Evening rush
            "2025-01-15 23:00:00",  # Late night
        ]
        
        results = []
        for time_str in test_times:
            revenue = self.calculate_expected_revenue(zone_id, time_str)
            results.append({
                'time': time_str,
                'time_display': time_str[-8:-3],  # Extract HH:MM
                'revenue': revenue
            })
        
        return results