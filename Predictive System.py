# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open("C:/Users/Victong/OneDrive/Desktop/vicp/phase_4_group_7/trained_model.sav", 'rb'))

# Sample input (make sure it matches the model's expected format)
input_data = (1200, 1, 2, 0, 0, 9, 9, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)


# Make prediction
prediction = loaded_model.predict(input_data_reshaped)

# If model outputs class labels directly:
predicted_class = prediction[0]

# If model outputs probabilities (comment this out unless needed)
# predicted_class = np.argmax(prediction[0])

# Map class index to label
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

# Show result
if 0 <= predicted_class < len(labels):
    print("Predicted Class:", labels[predicted_class])
else:
    print("Predicted Class Index is out of range.")
