import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import timedelta

# Set page config for a professional look
st.set_page_config(page_title="AQI Prediction System", layout="wide", page_icon="🌍")

# Title and description
st.title("🌿 AQI Prediction System")
st.markdown("""
    This app uses an Autoregressive (AR) model trained on historical AQI data from Kurla, Mumbai (MPCB).
    You can explore Kurla’s historical trends or input pollutant values for any location to predict the next day's AQI.
""")

# Sidebar for user inputs
st.sidebar.header("Settings")
lags = st.sidebar.slider("Select AR Model Lags", min_value=1, max_value=30, value=7, help="Number of past days used for prediction")
forecast_days = st.sidebar.number_input("Forecast Days (for Kurla)", min_value=1, max_value=7, value=1, help="Number of days to forecast for Kurla")

# Load and preprocess Kurla data
@st.cache_data
def load_and_preprocess_data(file_path='KurlaMumbaiMPCB.csv'):
    data = pd.read_csv(file_path)
    data['From Date'] = pd.to_datetime(data['From Date'])

    # AQI Calculation (CPCB Standard)
    def calculate_aqi(pm25, pm10, no2, so2, co, o3):
        aqi_pm25 = aqi_pm10 = aqi_no2 = aqi_so2 = aqi_co = aqi_o3 = 0
        if not pd.isna(pm25):
            if pm25 <= 30: aqi_pm25 = (50 / 30) * pm25
            elif pm25 <= 60: aqi_pm25 = 50 + (50 / 30) * (pm25 - 30)
            elif pm25 <= 90: aqi_pm25 = 100 + (100 / 30) * (pm25 - 60)
            elif pm25 <= 120: aqi_pm25 = 200 + (100 / 30) * (pm25 - 90)
            elif pm25 <= 250: aqi_pm25 = 300 + (100 / 130) * (pm25 - 120)
            elif pm25 > 250: aqi_pm25 = 400 + (100 / 250) * (pm25 - 250)
        if not pd.isna(pm10):
            if pm10 <= 50: aqi_pm10 = (50 / 50) * pm10
            elif pm10 <= 100: aqi_pm10 = 50 + (50 / 50) * (pm10 - 50)
            elif pm10 <= 250: aqi_pm10 = 100 + (100 / 150) * (pm10 - 100)
            elif pm10 <= 350: aqi_pm10 = 200 + (100 / 100) * (pm10 - 250)
            elif pm10 <= 430: aqi_pm10 = 300 + (100 / 80) * (pm10 - 350)
            elif pm10 > 430: aqi_pm10 = 400 + (100 / 430) * (pm10 - 430)
        if not pd.isna(no2):
            if no2 <= 40: aqi_no2 = (50 / 40) * no2
            elif no2 <= 80: aqi_no2 = 50 + (50 / 40) * (no2 - 40)
            elif no2 <= 180: aqi_no2 = 100 + (100 / 100) * (no2 - 80)
            elif no2 <= 280: aqi_no2 = 200 + (100 / 100) * (no2 - 180)
            elif no2 <= 400: aqi_no2 = 300 + (100 / 120) * (no2 - 280)
            elif no2 > 400: aqi_no2 = 400 + (100 / 400) * (no2 - 400)
        if not pd.isna(so2):
            if so2 <= 40: aqi_so2 = (50 / 40) * so2
            elif so2 <= 80: aqi_so2 = 50 + (50 / 40) * (so2 - 80)
            elif so2 <= 380: aqi_so2 = 100 + (100 / 300) * (so2 - 80)
            elif so2 <= 800: aqi_so2 = 200 + (100 / 420) * (so2 - 380)
            elif so2 <= 1600: aqi_so2 = 300 + (100 / 800) * (so2 - 800)
            elif so2 > 1600: aqi_so2 = 400 + (100 / 1600) * (so2 - 1600)
        if not pd.isna(co):
            if co <= 1: aqi_co = (50 / 1) * co
            elif co <= 2: aqi_co = 50 + (50 / 1) * (co - 1)
            elif co <= 10: aqi_co = 100 + (100 / 8) * (co - 2)
            elif co <= 17: aqi_co = 200 + (100 / 7) * (co - 10)
            elif co <= 34: aqi_co = 300 + (100 / 17) * (co - 17)
            elif co > 34: aqi_co = 400 + (100 / 34) * (co - 34)
        if not pd.isna(o3):
            if o3 <= 50: aqi_o3 = (50 / 50) * o3
            elif o3 <= 100: aqi_o3 = 50 + (50 / 50) * (o3 - 50)
            elif o3 <= 168: aqi_o3 = 100 + (100 / 68) * (o3 - 100)
            elif o3 <= 208: aqi_o3 = 200 + (100 / 40) * (o3 - 168)
            elif o3 <= 748: aqi_o3 = 300 + (100 / 540) * (o3 - 208)
            elif o3 > 748: aqi_o3 = 400 + (100 / 748) * (o3 - 748)
        return max(aqi_pm25, aqi_pm10, aqi_no2, aqi_so2, aqi_co, aqi_o3)

    data['AQI'] = data.apply(lambda row: calculate_aqi(row['PM2.5'], row['PM10'], row['NO2'], 
                                                      row['SO2'], row['CO'], row['Ozone']), axis=1)
    daily_data = data.resample('D', on='From Date').mean(numeric_only=True).reset_index()
    daily_data['AQI'] = daily_data['AQI'].interpolate()
    return data, daily_data

# Train AR model on Kurla data
@st.cache_resource
def train_ar_model(daily_data, lags):
    aqi_series = daily_data['AQI'].dropna()
    model = AutoReg(aqi_series, lags=lags).fit()
    return model

# Main app layout
data, daily_data = load_and_preprocess_data()
model = train_ar_model(daily_data, lags)

# Tabs for Kurla forecast and custom location prediction
tab1, tab2 = st.tabs(["Kurla AQI Forecast", "Custom Location AQI Prediction"])

# Tab 1: Kurla Forecast
with tab1:
    st.subheader("Kurla Data Overview")
    st.write("Snapshot of the daily AQI data for Kurla, Mumbai:")
    st.dataframe(daily_data[['From Date', 'AQI']].tail(10), use_container_width=True)

    # Historical AQI Plot
    st.subheader("Historical AQI Trend (Kurla)")
    fig = px.line(daily_data, x='From Date', y='AQI', title="Historical AQI (Kurla, Mumbai)",
                  labels={'AQI': 'Air Quality Index'}, template="plotly_dark")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Forecast for Kurla
    forecast = model.predict(start=len(daily_data['AQI']), end=len(daily_data['AQI']) + forecast_days - 1)
    last_date = daily_data['From Date'].iloc[-1]
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    st.subheader(f"AQI Forecast for Kurla - Next {forecast_days} Day(s)")
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted AQI': forecast})
    st.dataframe(forecast_df.style.format({"Forecasted AQI": "{:.2f}"}), use_container_width=True)

# Tab 2: Custom Location Prediction
with tab2:
    st.subheader("Predict AQI for a Custom Location")
    st.markdown("Enter pollutant values for a specific location to predict the next day's AQI using the Kurla-trained AR model.")

    # User inputs for pollutants
    col1, col2 = st.columns(2)
    with col1:
        pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=0.0)
        pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=0.0)
        no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, value=0.0)
    with col2:
        so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, value=0.0)
        co = st.number_input("CO (mg/m³)", min_value=0.0, value=0.0)
        o3 = st.number_input("Ozone (µg/m³)", min_value=0.0, value=0.0)

    if st.button("Predict AQI"):
        # Calculate current AQI for the input values
        current_aqi = calculate_aqi(pm25, pm10, no2, so2, co, o3)

        # Use the last `lags` values from Kurla data and append the new AQI
        recent_aqi = daily_data['AQI'].dropna().tail(lags).tolist()
        recent_aqi.append(current_aqi)
        recent_series = pd.Series(recent_aqi)

        # Predict the next day's AQI using the trained model
        next_day_aqi = model.predict(start=len(recent_series) - 1, end=len(recent_series) - 1, exog=recent_series[-lags:])[0]

        st.write(f"**Current AQI (based on inputs):** {current_aqi:.2f}")
        st.write(f"**Predicted AQI for Tomorrow:** {next_day_aqi:.2f}")

# Model Details
st.subheader("Model Details (Trained on Kurla Data)")
st.write(f"**Model Type**: Autoregressive (AR)")
st.write(f"**Lags Used**: {lags}")
st.write(f"**AIC**: {model.aic:.2f}")
st.write("Model Summary:")
st.text(str(model.summary()))

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center;'>Developed with ❤️ using Streamlit | Kurla Data Source: MPCB</p>
""", unsafe_allow_html=True)