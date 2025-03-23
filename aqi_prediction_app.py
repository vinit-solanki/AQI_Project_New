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
st.title("🌿 AQI Prediction System - Kurla, Mumbai")
st.markdown("""
    Welcome to the Air Quality Index (AQI) Prediction System for Kurla, Mumbai. This app uses an Autoregressive (AR) model to forecast AQI based on historical data from MPCB.
    Explore the data, visualize trends, and get predictions for the next day!
""")

# Sidebar for user inputs
st.sidebar.header("Settings")
lags = st.sidebar.slider("Select AR Model Lags", min_value=1, max_value=30, value=7, help="Number of past days used for prediction")
forecast_days = st.sidebar.number_input("Forecast Days", min_value=1, max_value=7, value=1, help="Number of days to forecast")

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data(file_path='KurlaMumbaiMPCB.csv'):
    # Load data
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

    # Calculate AQI
    data['AQI'] = data.apply(lambda row: calculate_aqi(row['PM2.5'], row['PM10'], row['NO2'], 
                                                      row['SO2'], row['CO'], row['Ozone']), axis=1)

    # Aggregate to daily data
    daily_data = data.resample('D', on='From Date').mean(numeric_only=True).reset_index()
    daily_data['AQI'] = daily_data['AQI'].interpolate()

    return data, daily_data

# Train AR model and forecast
@st.cache_resource
def train_ar_model(daily_data, lags, forecast_days):
    aqi_series = daily_data['AQI'].dropna()
    model = AutoReg(aqi_series, lags=lags).fit()
    forecast = model.predict(start=len(aqi_series), end=len(aqi_series) + forecast_days - 1)
    return model, forecast

# Main app layout
data, daily_data = load_and_preprocess_data()
st.subheader("Data Overview")
st.write("Snapshot of the daily AQI data:")
st.dataframe(daily_data[['From Date', 'AQI']].tail(10), use_container_width=True)

# Historical AQI Plot
st.subheader("Historical AQI Trend")
fig = px.line(daily_data, x='From Date', y='AQI', title="Historical AQI (Kurla, Mumbai)",
              labels={'AQI': 'Air Quality Index'}, template="plotly_dark")
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# Train model and forecast
model, forecast = train_ar_model(daily_data, lags, forecast_days)
last_date = daily_data['From Date'].iloc[-1]
forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]

# Daily Forecast
st.subheader(f"AQI Forecast for Next {forecast_days} Day(s)")
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted AQI': forecast})
st.dataframe(forecast_df.style.format({"Forecasted AQI": "{:.2f}"}), use_container_width=True)

# 15-Minute Forecast for Next Day
st.subheader("15-Minute AQI Forecast for Next Day")
intraday_pattern = data.groupby(data['From Date'].dt.time)['AQI'].mean()
intraday_pattern = intraday_pattern / intraday_pattern.mean()
next_day_start = last_date + timedelta(days=1)
next_day_dates = pd.date_range(start=next_day_start, periods=96, freq='15min')
next_day_forecast = forecast.iloc[0] * intraday_pattern.values
forecast_15min_df = pd.DataFrame({'Time': next_day_dates, 'Forecasted AQI': next_day_forecast})

fig_15min = px.line(forecast_15min_df, x='Time', y='Forecasted AQI', 
                    title=f"15-Minute AQI Forecast for {next_day_start.date()}",
                    template="plotly_dark")
st.plotly_chart(fig_15min, use_container_width=True)

# Model Details
st.subheader("Model Details")
st.write(f"**Model Type**: Autoregressive (AR)")
st.write(f"**Lags Used**: {lags}")
st.write(f"**AIC**: {model.aic:.2f}")
st.write("Model Summary:")
st.text(str(model.summary()))

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center;'>Developed with ❤️ using Streamlit | Data Source: MPCB</p>
""", unsafe_allow_html=True)