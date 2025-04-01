import pandas as pd
import numpy as np

def preprocess_timeseries(series, resample_rule='1H'):
    """Common preprocessing function for time series data"""
    try:
        # Handle missing values through interpolation
        series_clean = series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        
        # Remove outliers using IQR method with more robust bounds
        Q1 = series_clean.quantile(0.25)
        Q3 = series_clean.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2 * IQR  # More lenient lower bound
        upper_bound = Q3 + 2 * IQR  # More lenient upper bound
        series_clean = series_clean.clip(lower_bound, upper_bound)
        
        # Apply rolling mean to smooth the data before resampling
        window_size = 3  # Small window to preserve patterns while removing noise
        series_smooth = series_clean.rolling(
            window=window_size,
            center=True,
            min_periods=1
        ).mean()
        
        # Resample to reduce noise and redundant lines
        series_resampled = series_smooth.resample(resample_rule).mean()
        
        # Use linear interpolation instead of cubic for better stability
        series_resampled = series_resampled.interpolate(method='linear')
        
        # Fill any remaining NaN values
        series_resampled = series_resampled.fillna(method='bfill').fillna(method='ffill')
        
        return series_resampled
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return series  # Return original series if preprocessing fails

def load_data(filepath="../../datasets/KurlaMumbaiMPCB.csv"):
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv(filepath)
        df['From Date'] = pd.to_datetime(df['From Date'])
        df.set_index('From Date', inplace=True)
        return df
    except Exception as e:
        return None

def filter_time_range(df, time_range):
    """Filter dataframe based on time range"""
    end_date = df.index.max()
    
    if time_range == "1M":  # Last month
        start_date = end_date - pd.DateOffset(months=1)
    elif time_range == "3M":  # Last 3 months
        start_date = end_date - pd.DateOffset(months=3)
    elif time_range == "6M":  # Last 6 months
        start_date = end_date - pd.DateOffset(months=6)
    elif time_range == "1Y":  # Last year
        start_date = end_date - pd.DateOffset(years=1)
    elif time_range == "ALL":  # All data
        return df
    else:  # Default to 1 week
        start_date = end_date - pd.DateOffset(weeks=1)
    
    return df[start_date:end_date]

# Constants
TIME_RANGES = {
    "1W": "1 Week",
    "1M": "1 Month",
    "3M": "3 Months",
    "6M": "6 Months",
    "1Y": "1 Year",
    "ALL": "All Time"
}

RESAMPLE_RULES = {
    "1H": "Hourly",
    "6H": "6 Hours",
    "12H": "12 Hours",
    "1D": "Daily",
    "1W": "Weekly"
}

POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone']

# Enhanced Theme Settings for Dark Mode
THEME = {
    'primary_color': '#00bfff',      # Deep sky blue
    'secondary_color': '#ffa500',    # Orange
    'background_color': '#1a1a1a',   # Dark background
    'grid_color': '#333333',        # Dark gray grid
    'text_color': '#ffffff',        # White text
    'plot_template': 'plotly_dark',  # Dark template
    'font_family': 'Arial, sans-serif',
    'title_font_size': 24,
    'axis_font_size': 12,
    'label_font_size': 10,
    'margins': dict(l=50, r=50, t=80, b=50),
    'paper_bgcolor': '#1a1a1a',     # Dark background for plot area
    'plot_bgcolor': '#1a1a1a',      # Dark background for plotting area
    'legend_font_color': '#ffffff', # White text for legend
    'axis_color': '#ffffff'         # White axis lines
}

PLOT_TEMPLATE = "plotly_white" 