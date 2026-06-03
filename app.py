import streamlit as st
import joblib
import pandas as pd

# load model and columns
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Customer Churn Predictor")
st.write("Fill in the customer details below to predict if they will churn.")

# input fields
age = st.number_input("Age", min_value=18, max_value=70, value=30)
subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
watch_hours = st.number_input("Total Watch Hours", min_value=0.0, value=50.0)
last_login_days = st.number_input("Days Since Last Login", min_value=0, value=10)
monthly_fee = st.selectbox("Monthly Fee ($)", [8.99, 13.99, 17.99])
number_of_profiles = st.number_input("Number of Profiles", min_value=1, max_value=10, value=1)
avg_watch_hours_per_day = st.number_input("Avg Watch Hours Per Day", min_value=0.0, max_value=24.0, value=1.5)
payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"])
favorite_genre = st.selectbox("Favorite Genre", ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Documentary"])
region = st.selectbox("Region", ["North", "South", "East", "West"])
device = st.selectbox("Device", ["Mobile", "Tablet", "Smart TV", "Laptop"])

# predict button
if st.button("Predict Churn"):
    input_data = pd.DataFrame([{
        "age": age,
        "subscription_type": subscription_type,
        "watch_hours": watch_hours,
        "last_login_days": last_login_days,
        "monthly_fee": monthly_fee,
        "number_of_profiles": number_of_profiles,
        "avg_watch_hours_per_day": avg_watch_hours_per_day,
        "payment_method": payment_method,
        "favorite_genre": favorite_genre,
        "region": region,
        "device": device
    }])

    # encode to match training columns
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to churn — {probability:.0%} probability")
    else:
        st.success(f"✅ This customer is likely to stay — {1 - probability:.0%} probability")