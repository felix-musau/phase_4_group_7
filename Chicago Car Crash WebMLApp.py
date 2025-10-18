# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 09:30:37 2025

@author: Victong
"""

import numpy as np
import pickle 
import streamlit as st



# Load the saved model
loaded_model = pickle.load(open("C:/Users/Victong/OneDrive/Desktop/vicp/phase_4_group_7/trained_model.sav", 'rb'))




def primary_cause_prediction(input_data):
    # Sample input (make sure it matches the model's expected format)
    
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
        return "Predicted Class:", labels[predicted_class]
    else:
        return "Predicted Class Index is out of range."
    


def main ():
    # giving title
    st.title("Chicago Car Crash WebMLApp")
    
    # getting input data from user
    																		
    
    
    postedSpeedLimit = st.text_input("Posted Speed Limit")
    trafficControlDevice = st.text_input("Traffic Control Device")
    deviceCondition = st.text_input("Device Condition")
    weatherCondition = st.text_input("Weather Condition")
    lightingCondition = st.text_input("Lighting Condition")
    trafficwayType = st.text_input("Trafficway Type")
    alignment = st.text_input("Road Alignment")
    damage = st.text_input("Vehicle Damage Level")
    crashType = st.text_input("Crash Type")
    datePoliceNotified = st.text_input("Date Police Notified")
    streetNo = st.text_input("Street Number")
    streetDirection = st.text_input("Street Direction")
    streetName = st.text_input("Street Name")
    numUnits = st.text_input("Number of Units Involved")
    mostSevereInjury = st.text_input("Most Severe Injury")
    year = st.text_input("Crash Year")
    locationKey = st.text_input("Location Key")
    
    # code for prediction 
    Crush = ""
    
    # creating a button for prediction
    if st.button("Car Crush Result"):
        Crush = primary_cause_prediction([postedSpeedLimit,trafficControlDevice, deviceCondition, weatherCondition,lightingCondition, 
                                       trafficwayType,  alignment,damage, crashType,  datePoliceNotified, streetNo, streetDirection,
                                       streetName,numUnits,mostSevereInjury, year,locationKey])
    st.success(Crush)   


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


