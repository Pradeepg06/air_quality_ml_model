import streamlit as st
import joblib
import pandas as pd

# Page config
st.set_page_config(page_title="Air Quality Predictor", layout="wide")

# 🌙 DARK UI + CUSTOM INPUT WIDTH + DIVIDER
st.markdown("""
<style>

/* Background */
body {
    background-color: #0e1117;
    color: white;
}

/* Center Title */
.title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #9aa0a6;
    margin-bottom: 20px;
}

/* Vertical Divider */
.divider {
    border-left: 2px solid #2c2f36;
    height: 500px;
    margin: auto;
}

/* Input box width */
.stNumberInput > div > div > input {
    width: 250px !important;
}

/* Button */
.stButton>button {
    background-color: #1f6feb;
    color: white;
    border-radius: 8px;
    padding: 8px 20px;
}

/* Info box */
.info-box {
    background-color: #1f2a38;
    padding: 12px;
    border-radius: 10px;
    color: #4dabf7;
}

</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("air_quality_model.pkl")
feature_columns = joblib.load("features.pkl")

# AQI logic
def get_aqi_status(pm):
    if pm <= 50:
        return "🟢 Good"
    elif pm <= 100:
        return "🟡 Moderate"
    elif pm <= 150:
        return "🟠 Unhealthy (Sensitive)"
    elif pm <= 200:
        return "🔴 Unhealthy"
    else:
        return "🟣 Very Unhealthy"

# Precautions
def get_precautions(status):
    if "Good" in status:
        return "Air quality is good. Enjoy outdoor activities 😊"
    elif "Moderate" in status:
        return "Limit prolonged outdoor exertion."
    elif "Sensitive" in status:
        return "Children & elderly should reduce outdoor exposure."
    elif "Unhealthy" in status:
        return "Avoid outdoor activities. Wear mask."
    else:
        return "Stay indoors. Use air purifier."

# 🔝 HEADER
st.markdown('<div class="title">🌍 Air Quality Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict PM2.5 levels using Machine Learning</div>', unsafe_allow_html=True)

st.divider()

# Layout with divider column
col1, col_mid, col2 = st.columns([1, 0.1, 1])

# ⬅️ LEFT SIDE INPUTS
with col1:
    st.subheader("Input Parameters")

    temperature = st.number_input("Temperature (°C)", placeholder="Enter a number")
    humidity = st.number_input("Humidity (%)", placeholder="Enter a number")
    pressure = st.number_input("Pressure (hPa)", placeholder="Enter a number")
    wind_speed = st.number_input("Wind Speed (m/s)", placeholder="Enter a number")
    hour = st.slider("Hour of Day", 0, 23)

    predict_btn = st.button("Predict")

# 🔥 CENTER DIVIDER
with col_mid:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ➡️ RIGHT SIDE OUTPUT
with col2:
    st.subheader("Prediction Output")

    if predict_btn:

        if temperature == 0 or humidity == 0 or pressure == 0 or wind_speed == 0:
            st.warning("Please enter all values")
        else:
            with st.spinner("Predicting..."):

                input_data = {
                    'Season': 2,
                    'PM_Station 3': 90,
                    'Dew Point (Fahrenheit)': -5,
                    'Temperature (Fahrenheit)': 32,
                    'PM10 concentration (ug/m^3)': 50,
                    'SO2 concentration (ug/m^3)': 10,
                    'NO2 concentration (ug/m^3)': 20,
                    'CO concentration (ug/m^3)': 500,
                    'O3 concentration (ug/m^3)': 40,
                    'Pressure (hPa)': pressure,
                    'Humidity (%)': humidity,
                    'Wind Direction': 5,
                    'Wind Speed (m/s)': wind_speed,
                    'Precipitation (mm, hourly)': 0,
                    'Precipitation (mm, Cumulated)': 0,
                    'PM_Station 1': 90,
                    'PM_US Post': 90,
                    'PM_Station 2': 90,
                    'Dew Point (Celsius)': temperature - 5,
                    'Temperature (Celsius)': temperature,
                    'hour': hour,
                    'day': 15,
                    'month': 6
                }

                input_df = pd.DataFrame([input_data])
                input_df = input_df.reindex(columns=feature_columns)

                prediction = model.predict(input_df)[0]
                status = get_aqi_status(prediction)
                precautions = get_precautions(status)

            # RESULT DISPLAY
            st.success(f"PM2.5: {prediction:.2f} µg/m³")
            st.write(f"**Air Quality:** {status}")

            st.markdown("### ⚠️ Precautions")
            st.write(precautions)

    else:
        st.markdown('<div class="info-box">Enter values on the left and click Predict</div>', unsafe_allow_html=True)

st.divider()

# 🔻 FOOTER
st.markdown("""
<div style='text-align: center; color: gray; font-size: 14px;'>
📧 Contact: pradeepg.cs24@bmsce.ac.in | 🌐 GitHub: https://github.com/Pradeepg06/air_quality_ml_model.git
© 2026 Air Quality Predictor
</div>
""", unsafe_allow_html=True)