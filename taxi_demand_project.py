# Let's start - create a new Python file: taxi_demand_project.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# First, let's simulate what "wrong" looks like
# (We'll use real NYC data soon, but let's see the concept first)

# Imagine this is your data:
data = pd.DataFrame({
    'timestamp': pd.date_range('2024-03-01', periods=24*7, freq='h'),
    'zone': 'TimesSquare',
    'pickups': np.random.poisson(100, 24*7)  # Random pickup counts
})

print(data.head())

target_time = pd.Timestamp('2024-03-03 15:00')

# Calculate the average pickups in the last 3 hours before target_time
mask = (data['timestamp'] > (target_time - pd.Timedelta(hours=3))) & (data['timestamp'] <= target_time)
avg_pickups_last_3h = data.loc[mask, 'pickups'].mean()
print(f"Average pickups in last 3 hours before {target_time}: {avg_pickups_last_3h}")
