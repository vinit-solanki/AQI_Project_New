import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller, kpss
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (preprocess_timeseries, load_data, filter_time_range, 
                  TIME_RANGES, RESAMPLE_RULES, POLLUTANTS, THEME)

def perform_adf_test(series):
    """Perform Augmented Dickey-Fuller test"""
    series_clean = preprocess_timeseries(series)
    result = adfuller(series_clean)
    
    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05
    }

def perform_kpss_test(series):
    """Perform KPSS test"""
    series_clean = preprocess_timeseries(series)
    result = kpss(series_clean)
    
    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[3],
        'is_stationary': result[1] > 0.05
    }

def plot_transformations(series, resample_rule='1H'):
    """Plot original and transformed series"""
    # Preprocess the data
    series_clean = preprocess_timeseries(series, resample_rule)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Original Series", "Log Transformed",
                       "First Difference", "Seasonal Difference"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Original series
    fig.add_trace(
        go.Scatter(
            x=series_clean.index, 
            y=series_clean.values, 
            name="Original",
            line=dict(color=THEME['primary_color'], width=2)
        ),
        row=1, col=1
    )
    
    # Log transformation (add small constant to handle zeros)
    min_val = series_clean.min()
    if min_val <= 0:
        log_series = np.log1p(series_clean - min_val + 1)
    else:
        log_series = np.log1p(series_clean)
    
    fig.add_trace(
        go.Scatter(
            x=series_clean.index, 
            y=log_series.values, 
            name="Log",
            line=dict(color=THEME['secondary_color'], width=2)
        ),
        row=1, col=2
    )
    
    # First difference
    diff_series = series_clean.diff().dropna()
    fig.add_trace(
        go.Scatter(
            x=diff_series.index, 
            y=diff_series.values, 
            name="First Diff",
            line=dict(color='#2ecc71', width=2)
        ),
        row=2, col=1
    )
    
    # Seasonal difference (24 hours)
    seasonal_diff = series_clean.diff(24).dropna()
    fig.add_trace(
        go.Scatter(
            x=seasonal_diff.index, 
            y=seasonal_diff.values, 
            name="Seasonal Diff",
            line=dict(color='#9b59b6', width=2)
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Time Series Transformations",
        showlegend=True,
        template=THEME['plot_template'],
        plot_bgcolor=THEME['background_color'],
        paper_bgcolor=THEME['background_color']
    )
    
    # Update axes labels and grid
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(
                title_text="Date" if i == 2 else "",
                row=i, col=j,
                showgrid=True,
                gridwidth=1,
                gridcolor=THEME['grid_color']
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=THEME['grid_color']
            )
    
    return fig

def stationarity_page():
    st.title("Stationarity Analysis")
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stSelectbox {
            background-color: #ffffff;
            border-radius: 5px;
            padding: 5px;
        }
        .stSlider {
            padding-top: 10px;
            padding-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.write("""
    ### Understanding Stationarity
    A time series is considered stationary if its statistical properties (mean, variance) 
    remain constant over time. This is important for many time series modeling techniques.
    """)
    
    df = load_data()
    if df is not None:
        # Create three columns for controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pollutant = st.selectbox(
                "Select Pollutant",
                POLLUTANTS,
                help="Choose the air quality parameter to analyze"
            )
        
        with col2:
            time_range = st.selectbox(
                "Select Time Range",
                list(TIME_RANGES.keys()),
                format_func=lambda x: TIME_RANGES[x],
                index=1,
                help="Choose the time period for analysis"
            )
        
        with col3:
            resample_rule = st.selectbox(
                "Select Data Frequency",
                list(RESAMPLE_RULES.keys()),
                format_func=lambda x: RESAMPLE_RULES[x],
                help="Choose how to aggregate the data"
            )
        
        # Filter data based on time range
        filtered_df = filter_time_range(df, time_range)
        series = filtered_df[pollutant]
        
        # Show transformations
        st.subheader("Time Series Transformations")
        fig = plot_transformations(series, resample_rule)
        st.plotly_chart(fig, use_container_width=True)
        
        # Stationarity Tests
        st.subheader("Stationarity Tests")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Augmented Dickey-Fuller Test")
            adf_result = perform_adf_test(series)
            st.write(f"Test Statistic: {adf_result['test_statistic']:.4f}")
            st.write(f"P-value: {adf_result['p_value']:.4f}")
            st.write("Critical Values:")
            for key, value in adf_result['critical_values'].items():
                st.write(f"{key}: {value:.4f}")
            st.write(f"Series is {'stationary' if adf_result['is_stationary'] else 'non-stationary'}")
        
        with col2:
            st.markdown("#### KPSS Test")
            kpss_result = perform_kpss_test(series)
            st.write(f"Test Statistic: {kpss_result['test_statistic']:.4f}")
            st.write(f"P-value: {kpss_result['p_value']:.4f}")
            st.write("Critical Values:")
            for key, value in kpss_result['critical_values'].items():
                st.write(f"{key}: {value:.4f}")
            st.write(f"Series is {'stationary' if kpss_result['is_stationary'] else 'non-stationary'}")

if __name__ == "__main__":
    stationarity_page() 