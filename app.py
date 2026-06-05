import streamlit as st
import joblib
import pandas as pd

# page config
st.set_page_config(
    page_title="Netflix Churn Predictor",
    page_icon="🎬",
    layout="centered"
)

# custom CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 2rem; }
    .stButton>button {
        background-color: #E50914;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px 40px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background-color: #b20710; }
    h1 { color: #E50914; }
    h3 { color: #ffffff; border-bottom: 2px solid #E50914; padding-bottom: 8px; }
    </style>
""", unsafe_allow_html=True)

# load model and columns
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# header
st.markdown("<h1 style='text-align: center;'>🎬 Netflix Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Fill in the customer details below to predict if they will churn</p>", unsafe_allow_html=True)
st.markdown("---")

# section 1 - customer profile
st.markdown("### 👤 Customer Profile")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
with col2:
    subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
with col3:
    monthly_fee = st.selectbox("Monthly Fee ($)", [8.99, 13.99, 17.99])

st.markdown("---")

# section 2 - viewing behaviour
st.markdown("### 📺 Viewing Behaviour")
col4, col5, col6 = st.columns(3)
with col4:
    watch_hours = st.number_input("Total Watch Hours", min_value=0.0, value=50.0)
with col5:
    avg_watch_hours_per_day = st.number_input("Avg Watch Hours Per Day", min_value=0.0, max_value=24.0, value=1.5)
with col6:
    favorite_genre = st.selectbox("Favorite Genre", ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Documentary"])

st.markdown("---")

# section 3 - account details
st.markdown("### 🔐 Account Details")
col7, col8, col9 = st.columns(3)
with col7:
    last_login_days = st.number_input("Days Since Last Login", min_value=0, value=10)
with col8:
    number_of_profiles = st.number_input("Number of Profiles", min_value=1, max_value=10, value=1)
with col9:
    payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"])

st.markdown("---")

# section 4 - location and device
st.markdown("### 🌍 Location & Device")
col10, col11 = st.columns(2)
with col10:
    region = st.selectbox("Region", ["Africa", "Asia", "Europe", "North America", "Oceania", "North America"])
with col11:
    device = st.selectbox("Device", ["Mobile", "Tablet", "Smart TV", "Laptop"])

st.markdown("---")

# predict button
if st.button("🔮 Predict Churn"):
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

    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    st.markdown("---")
    st.markdown("### 📊 Prediction Result")

    col_a, col_b, col_c = st.columns(3)
    with col_b:
        if prediction == 1:
            st.error(f"⚠️ Likely to Churn")
            st.metric(label="Churn Probability", value=f"{probability:.0%}")
        else:
            st.success(f"✅ Likely to Stay")
            st.metric(label="Retention Probability", value=f"{1 - probability:.0%}")
