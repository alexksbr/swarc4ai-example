import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

from .constants import (
    CITY_AVG_FARE, CITY_AVG_TRIP_TIME_MINUTES, AIRPORT_AVG_FARE, 
    AIRPORT_AVG_TRIP_TIME_MINUTES, AIRPORT_PASSENGER_TAXI_RATE,
    INTERNATIONAL_FLIGHT_MULTIPLIER, DEMAND_NORMALIZATION_CITY,
    DEMAND_NORMALIZATION_AIRPORT, DAYTIME_FLIGHT_RANGE, NIGHTTIME_FLIGHT_RANGE,
    DAYTIME_HOURS, FLIGHT_ARRIVAL_DELAY_RANGE, PASSENGER_RANGE,
    INTERNATIONAL_FLIGHT_PROBABILITY, MINUTES_PER_HOUR
)
from .utils import validate_zone_id, safe_divide, ensure_non_negative

logger = logging.getLogger(__name__)

class AirportDemandPredictor:
    """
    Predictor for comparing airport vs city demand for taxi drivers.
    
    This class helps taxi drivers decide whether to stay in their current zone
    or head to the airport based on demand and revenue predictions.
    """
    
    def __init__(
        self, 
        taxi_feature_store: Any, 
        wait_time_predictor: Optional[Any] = None
    ) -> None:
        """
        Initialize the airport demand predictor.
        
        Args:
            taxi_feature_store: Instance of TaxiFeatureStore
            wait_time_predictor: Optional trained WaitTimePredictor instance
        """
        self.taxi_store = taxi_feature_store
        self.wait_time_predictor = wait_time_predictor
        
    def generate_flight_arrivals(self, current_time: datetime) -> List[Dict[str, Union[datetime, int, bool]]]:
        """
        Generate simulated flight arrivals based on time of day.
        
        Args:
            current_time: Current datetime for simulation
            
        Returns:
            List of flight dictionaries with arrival times and passenger counts
        """
        try:
            hour = current_time.hour
            
            # More flights during day (6AM-10PM)
            if DAYTIME_HOURS[0] <= hour <= DAYTIME_HOURS[1]:
                num_flights = random.randint(*DAYTIME_FLIGHT_RANGE)
            else:
                num_flights = random.randint(*NIGHTTIME_FLIGHT_RANGE)
                
            flights = []
            for _ in range(num_flights):
                flights.append({
                    'arrival_time': current_time + timedelta(
                        minutes=random.randint(*FLIGHT_ARRIVAL_DELAY_RANGE)
                    ),
                    'passengers': random.randint(*PASSENGER_RANGE),
                    'international': random.random() < INTERNATIONAL_FLIGHT_PROBABILITY,
                })
            return flights
            
        except Exception as e:
            logger.error(f"Error generating flight arrivals: {e}")
            return []
    
    def estimate_airport_demand(self, current_time: datetime) -> int:
        """
        Estimate taxis needed at airport in the next hour.
        
        Args:
            current_time: Current datetime for estimation
            
        Returns:
            Estimated number of taxi rides needed at airport
        """
        try:
            flights = self.generate_flight_arrivals(current_time)
            
            taxi_demand = 0.0
            for flight in flights:
                # Assume 30% of passengers take taxis
                taxi_passengers = flight['passengers'] * AIRPORT_PASSENGER_TAXI_RATE
                
                # International flights = more taxi usage
                if flight['international']:
                    taxi_passengers *= INTERNATIONAL_FLIGHT_MULTIPLIER
                    
                taxi_demand += taxi_passengers
                
            return ensure_non_negative(int(taxi_demand))
            
        except Exception as e:
            logger.error(f"Error estimating airport demand: {e}")
            return 0
    
    def _get_city_demand_prediction(self, zone_id: int, current_time: datetime) -> int:
        """
        Get city demand prediction using the trained model or fallback to historical data.
        
        Args:
            zone_id: Zone ID for prediction
            current_time: Current datetime
            
        Returns:
            Predicted demand for the zone
            
        Raises:
            ValueError: If zone_id is invalid
        """
        if not validate_zone_id(zone_id):
            raise ValueError(f"Invalid zone ID: {zone_id}")
        
        if self.wait_time_predictor and hasattr(self.wait_time_predictor, '_is_trained') and self.wait_time_predictor._is_trained:
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
                return ensure_non_negative(int(prediction))
            except Exception as e:
                # Fallback to historical data if model prediction fails
                logger.warning(f"Model prediction failed ({e}), falling back to historical data")
                return self.taxi_store.calculate_next_hour_pickups(zone_id, current_time)
        else:
            # Fallback to historical data if no trained model available
            logger.info("No trained model available, using historical data")
            return self.taxi_store.calculate_next_hour_pickups(zone_id, current_time)
    
    def compare_locations(self, zone_id: int, current_time: datetime) -> Dict[str, Union[int, str]]:
        """
        Compare city zone vs airport demand to make recommendation.
        
        Args:
            zone_id: Current zone ID
            current_time: Current datetime
            
        Returns:
            Dictionary with comparison results and recommendation
        """
        try:
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
            
        except Exception as e:
            logger.error(f"Error comparing locations for zone {zone_id}: {e}")
            return {
                'city_zone': zone_id,
                'city_demand': 0,
                'airport_demand': 0,
                'recommendation': 'STAY',
                'error': str(e)
            }
    
    def calculate_expected_revenue(
        self, 
        zone_id: int, 
        current_time: Union[str, datetime]
    ) -> Dict[str, Union[int, float, str]]:
        """
        Calculate expected revenue per hour for city vs airport.
        
        Args:
            zone_id: Current zone ID
            current_time: Current time as datetime or string
            
        Returns:
            Dictionary with revenue calculations and recommendation
        """
        try:
            # Parse current_time if it's a string
            if isinstance(current_time, str):
                current_time = datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
            
            # Get demand estimates (using model inference)
            city_demand = self._get_city_demand_prediction(zone_id, current_time)
            airport_demand = self.estimate_airport_demand(current_time)
            
            # Maximum trips limited by time
            max_city_trips_per_hour = safe_divide(MINUTES_PER_HOUR, CITY_AVG_TRIP_TIME_MINUTES)
            max_airport_trips_per_hour = safe_divide(MINUTES_PER_HOUR, AIRPORT_AVG_TRIP_TIME_MINUTES)
            
            # Calculate actual trips per hour based on demand and time constraints
            # Higher demand means higher probability of getting rides quickly
            city_demand_factor = min(1.0, city_demand / DEMAND_NORMALIZATION_CITY)
            airport_demand_factor = min(1.0, airport_demand / DEMAND_NORMALIZATION_AIRPORT)
            
            city_trips_per_hour = min(max_city_trips_per_hour, city_demand_factor * max_city_trips_per_hour)
            airport_trips_per_hour = min(max_airport_trips_per_hour, airport_demand_factor * max_airport_trips_per_hour)
            
            # Calculate hourly revenue
            city_revenue_per_hour = city_trips_per_hour * CITY_AVG_FARE
            airport_revenue_per_hour = airport_trips_per_hour * AIRPORT_AVG_FARE
            
            return {
                'city_zone': zone_id,
                'city_demand': city_demand,
                'city_trips_per_hour': city_trips_per_hour,
                'city_avg_fare': CITY_AVG_FARE,
                'city_revenue_per_hour': city_revenue_per_hour,
                'airport_demand': airport_demand,
                'airport_trips_per_hour': airport_trips_per_hour,
                'airport_avg_fare': AIRPORT_AVG_FARE,
                'airport_revenue_per_hour': airport_revenue_per_hour,
                'revenue_recommendation': 'AIRPORT' if airport_revenue_per_hour > city_revenue_per_hour else 'STAY',
                'revenue_difference': abs(airport_revenue_per_hour - city_revenue_per_hour)
            }
            
        except Exception as e:
            logger.error(f"Error calculating expected revenue for zone {zone_id}: {e}")
            return {
                'city_zone': zone_id,
                'city_demand': 0,
                'city_trips_per_hour': 0.0,
                'city_avg_fare': CITY_AVG_FARE,
                'city_revenue_per_hour': 0.0,
                'airport_demand': 0,
                'airport_trips_per_hour': 0.0,
                'airport_avg_fare': AIRPORT_AVG_FARE,
                'airport_revenue_per_hour': 0.0,
                'revenue_recommendation': 'STAY',
                'revenue_difference': 0.0,
                'error': str(e)
            }
    
    def test_revenue_recommendations(self, zone_id: int = 161) -> List[Dict[str, Union[str, Dict]]]:
        """
        Test revenue model at different times of day.
        
        Args:
            zone_id: Zone ID to test (default: 161 - Times Square)
            
        Returns:
            List of revenue test results for different times
        """
        if not validate_zone_id(zone_id):
            raise ValueError(f"Invalid zone ID: {zone_id}")
        
        test_times = [
            "2025-01-15 06:00:00",  # Early morning
            "2025-01-15 09:00:00",  # Morning rush
            "2025-01-15 14:00:00",  # Afternoon
            "2025-01-15 18:00:00",  # Evening rush
            "2025-01-15 23:00:00",  # Late night
        ]
        
        results = []
        for time_str in test_times:
            try:
                revenue = self.calculate_expected_revenue(zone_id, time_str)
                results.append({
                    'time': time_str,
                    'time_display': time_str[-8:-3],  # Extract HH:MM
                    'revenue': revenue
                })
            except Exception as e:
                logger.error(f"Error testing revenue for time {time_str}: {e}")
                results.append({
                    'time': time_str,
                    'time_display': time_str[-8:-3],
                    'revenue': {'error': str(e)}
                })
        
        return results