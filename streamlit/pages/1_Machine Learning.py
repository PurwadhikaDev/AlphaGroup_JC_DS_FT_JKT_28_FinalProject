import os
import sys
import joblib

sys.path.append("C:/Users/Dell/Documents/Purwadhika/Final Project/")

from custom_encoders import BinaryEncoderWrapper

import streamlit as st
import pandas as pd
import numpy as np
import gzip
import pickle
from category_encoders import BinaryEncoder
from sklearn.base import clone, BaseEstimator

# Page config
st.set_page_config(page_title="Cancellation Prediction", layout="wide")

# Title
st.title("🔍 Booking Cancellation Prediction")
st.markdown("""
This page allows you to test the cancellation prediction model.  
Please input booking details below and click **Predict**.
""")

# Load the model
@st.cache_resource
def load_model():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../model.joblib"))
    model = joblib.load(model_path)
    return model

model = load_model()

# Input layout
with st.form("prediction_form"):
    st.subheader("🧾 Booking Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        hotel = st.selectbox("Hotel Type", ['City Hotel', 'Resort Hotel'])
        lead_time = st.number_input("Lead Time (days)", 0, 500, 30)
        adults = st.number_input("Number of Adults", 1, 4, 2)
        children = st.number_input("Number of Children", 0, 4, 0)
        babies = st.number_input("Number of Babies", 0, 4, 0)
        arrival_day = st.number_input("Arrival Day of Month", 1, 31, 15)
        arrival_week = st.number_input("Arrival Week Number", 1, 53, 26)
    
    with col2:
        arrival_month = st.selectbox("Arrival Month", [
            'January', 'February', 'March', 'April', 'May', 'June', 'July',
            'August', 'September', 'October', 'November', 'December'
        ])
        weekend_nights = st.number_input("Weekend Nights", 0, 10, 1)
        week_nights = st.number_input("Week Nights", 0, 10, 2)
        deposit_type = st.selectbox("Deposit Type", ['No Deposit', 'Non Refund', 'Refundable'])
        customer_type = st.selectbox("Customer Type", ['Transient', 'Contract', 'Transient-Party', 'Group'])
        arrival_year = st.number_input("Arrival Year", 2015, 2018, 2017)
        waiting_list_days = st.number_input("Days in Waiting List", 0, 300, 0)
        parking_spaces = st.number_input("Required Parking Spaces", 0, 5, 0)

    with col3:
        is_repeated_guest = st.selectbox("Repeated Guest?", [0, 1])
        booking_changes = st.number_input("Booking Changes", 0, 10, 0)
        special_requests = st.number_input("Special Requests", 0, 5, 0)
        reserved_room = st.selectbox("Reserved Room Type", list('ABCDEFGHI'))
        meal = st.selectbox("Meal Type", ['BB', 'HB', 'FB', 'SC', 'Undefined'])
        previous_cancellations = st.number_input("Previous Cancellations", 0, 10, 0)
        previous_bookings = st.number_input("Previous Bookings Not Canceled", 0, 20, 0)
        adr = st.number_input("Average Daily Rate (ADR)", 0.0, 1000.0, 100.0, step=0.1)


    submitted = st.form_submit_button("Predict Cancellation")

# On predict
if submitted:
    input_data = pd.DataFrame([{
        'adr': adr,
        'hotel': hotel,
        'lead_time': lead_time,
        'adults': adults,
        'children': children,
        'babies': babies,
        'arrival_date_month': arrival_month,
        'arrival_date_day_of_month': arrival_day,
        'arrival_date_week_number': arrival_week,
        'arrival_date_year': arrival_year,
        'stays_in_weekend_nights': weekend_nights,
        'stays_in_week_nights': week_nights,
        'deposit_type': deposit_type,
        'customer_type': customer_type,
        'is_repeated_guest': is_repeated_guest,
        'booking_changes': booking_changes,
        'days_in_waiting_list': waiting_list_days,
        'required_car_parking_spaces': parking_spaces,
        'previous_cancellations': previous_cancellations,
        'previous_bookings_not_canceled': previous_bookings,
        'total_of_special_requests': special_requests,
        'reserved_room_type': reserved_room,
        'meal': meal,
        'distribution_channel': 'TA/TO',  # placeholder
        'market_segment': 'Online TA',    # placeholder
        'country': 'PRT',                 # placeholder
    }])


    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")
    st.write(f"**Prediction:** {'❌ Canceled' if prediction == 1 else '✅ Not Canceled'}")
    st.write(f"**Cancellation Probability:** {proba:.2%}")