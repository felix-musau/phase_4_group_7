# -*- coding: utf-8 -*-
"""
Chicago Car Crash Cause Predictor (Streamlit App)
"""

import numpy as np
import pickle
import streamlit as st

# Load the saved model
model_path = "C:/Users/Victong/OneDrive/Desktop/vicp/phase_4_group_7/trained_model.sav"
loaded_model = pickle.load(open(model_path, 'rb'))

# Prediction function
def primary_cause_prediction(input_data):
    try:
        input_array = np.asarray(input_data, dtype=np.float32).reshape(1, -1)
        prediction = loaded_model.predict(input_array)
        predicted_class = int(prediction[0])
        
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
        
        if 0 <= predicted_class < len(labels):
            return f"Predicted Crash Cause: **{labels[predicted_class]}**"
        else:
            return "Predicted class index is out of range."
    except Exception as e:
        return f"Prediction failed: {str(e)}"

# Streamlit App
def main():
    st.title("🚗 Chicago Car Crash Cause Predictor")
    st.markdown("Enter the crash details below to predict the **primary cause group**.")

    # Collect input for all 20 features
    posted_speed_limit = st.text_input("Posted Speed Limit")
    traffic_control_device = st.text_input("Traffic Control Device")
    device_condition = st.text_input("Device Condition")
    weather_condition = st.text_input("Weather Condition")
    lighting_condition = st.text_input("Lighting Condition")
    first_crash_type = st.text_input("First Crash Type")
    trafficway_type = st.text_input("Trafficway Type")
    alignment = st.text_input("Road Alignment")
    road_defect = st.text_input("Road Defect")
    crash_type = st.text_input("Crash Type")
    damage = st.text_input("Vehicle Damage Level")
    date_police_notified = st.text_input("Date Police Notified (e.g., 20251018)")
    sec_contributory_cause = st.text_input("Secondary Contributory Cause")
    street_no = st.text_input("Street Number")
    street_direction = st.text_input("Street Direction")
    street_name = st.text_input("Street Name")
    num_units = st.text_input("Number of Units Involved")
    most_severe_injury = st.text_input("Most Severe Injury")
    year = st.text_input("Crash Year")
    location_key = st.text_input("Location Key")

    Crush = ""

    if st.button("🚨 Predict Crash Cause"):
        try:
            # Convert all inputs to float
            input_features = [
                float(posted_speed_limit),
                float(traffic_control_device),
                float(device_condition),
                float(weather_condition),
                float(lighting_condition),
                float(first_crash_type),
                float(trafficway_type),
                float(alignment),
                float(road_defect),
                float(crash_type),
                float(damage),
                float(date_police_notified),
                float(sec_contributory_cause),
                float(street_no),
                float(street_direction),
                float(street_name),
                float(num_units),
                float(most_severe_injury),
                float(year),
                float(location_key),
            ]
            Crush = primary_cause_prediction(input_features)
            st.success(Crush)

        except ValueError:
            st.error("⚠️ Please enter valid numeric values in all fields.")

if __name__ == "__main__":
    main()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


