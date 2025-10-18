# -*- coding: utf-8 -*-
"""
Testing trained XGBoost crash cause model
"""

import numpy as np
import pickle

# Load the trained model
model_path = "C:/Users/Victong/OneDrive/Desktop/vicp/phase_4_group_7/trained_model.sav"
loaded_model = pickle.load(open(model_path, 'rb'))

# === Prepare numeric input in exact feature order ===
# Replace dummy values below with actual numeric input as per preprocessing
input_data = (
    30,     # posted_speed_limit
    1,      # traffic_control_device (e.g., 1 = Traffic Signal)
    0,      # device_condition
    1,      # weather_condition
    0,      # lighting_condition
    2,      # first_crash_type
    1,      # trafficway_type
    0,      # alignment
    0,      # road_defect
    2,      # crash_type
    1,      # damage
    20231017,  # date_police_notified (use int format e.g., YYYYMMDD)
    3,      # sec_contributory_cause
    100,    # street_no
    1,      # street_direction
    12,     # street_name
    2,      # num_units
    0,      # most_severe_injury
    2023,   # year
    9813    # Location_Key
)

# Convert to NumPy array and reshape
input_array = np.asarray(input_data, dtype=np.float32).reshape(1, -1)

# Make prediction
prediction = loaded_model.predict(input_array)
predicted_class = int(prediction[0])

# Map class index to human-readable label
labels = [
    'Speed & Right-of-Way Failure',
    'Maneuver & Positioning Error',
    'Disregard Traffic Controls',
    'Impairment & Recklessness',
    'Distraction & Visibility',
    'External/Environmental/Mechanical',
    'Unclassified/Administrative',
    'UNABLE TO DETERMINE'
]

# Output prediction
if 0 <= predicted_class < len(labels):
    print("Predicted Crash Cause:", labels[predicted_class])
else:
    print("Error: Predicted class index is out of range.")
