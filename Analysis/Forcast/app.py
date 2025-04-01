import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set page config
st.set_page_config(
    page_title="PM2.5 Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("kurla_daily.csv", parse_dates=["From Date"], index_col="From Date")
    return df

# Feature engineering function
def create_features(series, lags=7):
    df = pd.DataFrame(series)
    for lag in range(1, lags+1):
        df[f'lag_{lag}'] = df['PM2.5_filled'].shift(lag)
    
    df['rolling_mean_7'] = df['PM2.5_filled'].shift(1).rolling(7, min_periods=4).mean()
    df['rolling_std_7'] = df['PM2.5_filled'].shift(1).rolling(7, min_periods=4).std()
    
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df.index.dayofweek >= 5
    df['month'] = df.index.month
    df['diff_1'] = df['PM2.5_filled'].diff(1)
    
    df.dropna(inplace=True)
    return df

# Train models function
def train_models(train_data):
    # ARIMA Model
    arima_model = ARIMA(
        order=(1,1,1),
        seasonal_order=(1,0,1,25),
        suppress_warnings=True
    )
    arima_model.fit(train_data)
    
    # Get ARIMA predictions
    train_pred_arima = pd.Series(arima_model.predict_in_sample(), index=train_data.index)
    
    # Create features for residual model
    train_features = create_features(train_data)
    train_residuals = train_data.loc[train_features.index] - train_pred_arima.loc[train_features.index]
    
    # Prepare data for residual model
    train_features['arima_residual'] = train_residuals
    X_train = train_features.drop('PM2.5_filled', axis=1)
    y_train = train_features['arima_residual']
    
    # Train residual model
    residual_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=3,
        min_samples_leaf=10,
        max_features=0.5,
        random_state=42
    )
    residual_model.fit(X_train, y_train)
    
    return arima_model, residual_model

# Generate predictions function
def generate_predictions(arima_model, residual_model, last_data, periods):
    # Generate ARIMA predictions
    arima_pred = pd.Series(
        arima_model.predict(n_periods=periods),
        index=pd.date_range(start=last_data.index[-1] + pd.Timedelta(days=1), periods=periods, freq='D')
    )
    
    # Create features for residual predictions
    pred_features = create_features(pd.concat([last_data, arima_pred]))
    pred_features = pred_features.tail(periods)
    
    # Generate residual predictions
    X_pred = pred_features.drop('PM2.5_filled', axis=1)
    residual_pred = residual_model.predict(X_pred)
    
    # Combine predictions
    final_pred = arima_pred.loc[pred_features.index] + residual_pred
    
    return final_pred

# Main app
def main():
    st.title("PM2.5 Forecasting Dashboard")
    st.write("Select a page from the sidebar to view different prediction timeframes")
    
    # Load data
    df = load_data()
    pm25_series = df["PM2.5_filled"].dropna()
    
    # Display data info
    st.subheader("Dataset Information")
    st.write(f"Data range: {pm25_series.index.min()} to {pm25_series.index.max()}")
    st.write(f"Total observations: {len(pm25_series)}")
    
    # Plot historical data
    st.subheader("Historical PM2.5 Levels")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pm25_series)
    ax.set_title("Historical PM2.5 Levels")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (μg/m³)")
    st.pyplot(fig)

if __name__ == "__main__":
    main()