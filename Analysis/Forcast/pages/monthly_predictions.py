import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Page config
st.set_page_config(
    page_title="Monthly PM2.5 Predictions",
    page_icon="📈",
    layout="wide"
)

# Load and prepare data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("./kurla_daily.csv", parse_dates=["From Date"], index_col="From Date")
        if "PM2.5_filled" not in df.columns:
            raise KeyError("Required column 'PM2.5_filled' not found in the dataset")
        return df
    except FileNotFoundError:
        st.error("Error: Could not find kurla_daily.csv file")
        st.stop()
    except KeyError as e:
        st.error(f"Error: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data: {str(e)}")
        st.stop()

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

# Main page
st.title("Monthly PM2.5 Predictions")

# Load data
df = load_data()
pm25_series = df["PM2.5_filled"].dropna()

# Train models
train_data = pm25_series[:-30]  # Use all but last month for training
arima_model, residual_model = train_models(train_data)

# Generate monthly predictions
monthly_pred = generate_predictions(arima_model, residual_model, train_data, periods=30)

# Display predictions
st.subheader("Next Month's PM2.5 Forecasts")
col1, col2 = st.columns(2)

with col1:
    # Plot predictions
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pm25_series[-30:], label='Historical', color='blue')
    ax.plot(monthly_pred, label='Forecast', color='red')
    ax.fill_between(
        monthly_pred.index,
        monthly_pred * 0.9,
        monthly_pred * 1.1,
        color='red',
        alpha=0.2,
        label='90% Confidence'
    )
    ax.set_title("PM2.5 Monthly Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (μg/m³)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    # Display predictions table with weekly aggregates
    st.write("Weekly Aggregated Predictions")
    weekly_pred = pd.DataFrame({
        'Week': [f'Week {i+1}' for i in range(4)],
        'Average PM2.5': [
            monthly_pred[i*7:(i+1)*7].mean().round(2) for i in range(4)
        ],
        'Min PM2.5': [
            monthly_pred[i*7:(i+1)*7].min().round(2) for i in range(4)
        ],
        'Max PM2.5': [
            monthly_pred[i*7:(i+1)*7].max().round(2) for i in range(4)
        ]
    })
    st.dataframe(weekly_pred)

    # Display daily predictions
    st.write("\nDetailed Daily Predictions")
    pred_df = pd.DataFrame({
        'Date': monthly_pred.index.strftime('%Y-%m-%d'),
        'PM2.5 Forecast': monthly_pred.round(2),
        'Lower Bound': (monthly_pred * 0.9).round(2),
        'Upper Bound': (monthly_pred * 1.1).round(2)
    })
    st.dataframe(pred_df)

# Download predictions
st.download_button(
    label="Download Monthly Predictions",
    data=pred_df.to_csv(index=False),
    file_name="monthly_pm25_predictions.csv",
    mime="text/csv"
)

# Model performance metrics
st.subheader("Model Performance Metrics")

# Calculate metrics on recent data
recent_actual = pm25_series[-60:-30]  # Last 60 days excluding prediction period
recent_pred = generate_predictions(arima_model, residual_model, pm25_series[:-60], periods=30)

metrics = {
    'MAE': mean_absolute_error(recent_actual, recent_pred),
    'RMSE': np.sqrt(mean_squared_error(recent_actual, recent_pred)),
    'R²': r2_score(recent_actual, recent_pred)
}

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Mean Absolute Error", f"{metrics['MAE']:.2f} μg/m³")
with col2:
    st.metric("Root Mean Square Error", f"{metrics['RMSE']:.2f} μg/m³")
with col3:
    st.metric("R² Score", f"{metrics['R²']:.3f}")

# Additional visualizations
st.subheader("Forecast Analysis")
col1, col2 = st.columns(2)

with col1:
    # Monthly trend comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    monthly_avg = monthly_pred.groupby(monthly_pred.index.dayofweek).mean()
    historical_avg = pm25_series.groupby(pm25_series.index.dayofweek).mean()
    
    ax.plot(monthly_avg.index, monthly_avg.values, label='Forecast Trend', color='red')
    ax.plot(historical_avg.index, historical_avg.values, label='Historical Trend', color='blue')
    ax.set_title("Day-of-Week Pattern Comparison")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average PM2.5 (μg/m³)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    # Forecast distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    monthly_pred.hist(ax=ax, bins=20, color='skyblue', alpha=0.6)
    ax.set_title("Forecast Distribution")
    ax.set_xlabel("PM2.5 (μg/m³)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig)