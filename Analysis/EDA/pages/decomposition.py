import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (preprocess_timeseries, load_data, filter_time_range, 
                  TIME_RANGES, RESAMPLE_RULES, POLLUTANTS, THEME)

def plot_decomposition(series, period, model_type='additive', resample_rule='1H'):
    """Plot time series decomposition"""
    try:
        # Preprocess the data
        series_clean = preprocess_timeseries(series, resample_rule)
        
        # Apply smoothing using rolling mean to reduce noise
        series_smooth = series_clean.rolling(window=12, center=True, min_periods=1).mean()
        
        # Check if data contains zero or negative values for multiplicative decomposition
        if model_type == 'multiplicative' and (series_smooth <= 0).any():
            st.warning("Data contains zero or negative values. Switching to additive decomposition.")
            model_type = 'additive'
        
        # Validate period
        min_period = 4  # Minimum period for decomposition
        max_period = min(168, len(series_smooth) // 2)  # Maximum period (weekly or half data length)
        
        if period < min_period:
            st.warning(f"Seasonal period ({period}) is too small. Using minimum period of {min_period}.")
            period = min_period
        elif period > max_period:
            st.warning(f"Seasonal period ({period}) is too large. Using maximum period of {max_period}.")
            period = max_period
        
        # Ensure period is even for better decomposition
        if period % 2 != 0:
            period = period + 1
            st.info(f"Adjusted period to {period} for better decomposition.")
        
        # Perform decomposition
        decomposition = seasonal_decompose(
            series_smooth, 
            period=period, 
            model=model_type
        )
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=("Original", "Trend", "Seasonal", "Residual"),
            vertical_spacing=0.15,
            specs=[[{"secondary_y": False}]] * 4
        )
        
        # Plot original data with smoothed line
        fig.add_trace(
            go.Scatter(
                x=series_clean.index,
                y=series_clean.values,
                name="Raw Data",
                line=dict(color='lightgray', width=1),
                opacity=0.5,
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=series_smooth.index,
                y=series_smooth.values,
                name="Smoothed",
                line=dict(color=THEME['primary_color'], width=2),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Plot components
        components = [
            (decomposition.trend, "Trend", THEME['secondary_color']),
            (decomposition.seasonal, "Seasonal", '#2ecc71'),
            (decomposition.resid, "Residual", '#9b59b6')
        ]
        
        for idx, (component, name, color) in enumerate(components, 2):
            fig.add_trace(
                go.Scatter(
                    x=series_clean.index,
                    y=component,
                    name=name,
                    line=dict(color=color, width=2)
                ),
                row=idx, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text=f"{model_type.capitalize()} Decomposition (Period: {period})",
            showlegend=True,
            template="plotly_white",
            plot_bgcolor=THEME['background_color'],
            paper_bgcolor=THEME['background_color']
        )
        
        # Update axes labels and grid
        components = ["Original Value", "Trend", "Seasonal", "Residual"]
        for i, title in enumerate(components, 1):
            fig.update_xaxes(
                title_text="Date" if i == 4 else "",
                row=i, col=1,
                showgrid=True,
                gridwidth=1,
                gridcolor=THEME['grid_color']
            )
            fig.update_yaxes(
                title_text=title,
                row=i, col=1,
                showgrid=True,
                gridwidth=1,
                gridcolor=THEME['grid_color']
            )
        
        return fig
    except Exception as e:
        st.error(f"Error in decomposition: {str(e)}. Try adjusting the seasonal period or switching to additive decomposition.")
        return None

def decomposition_page():
    st.title("Time Series Decomposition Analysis")
    
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
    ### Understanding Time Series Components
    Time series decomposition helps us understand the underlying patterns in your air quality data by breaking it down into its components:
    - **Trend**: Long-term progression of the series
    - **Seasonality**: Regular patterns of ups and downs
    - **Residuals**: Random variations
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
                index=1,  # Default to 1 month
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
        
        # Decomposition controls
        st.subheader("Decomposition Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            # Calculate appropriate period range
            min_period = 4
            max_period = min(168, len(series) // 2)
            default_period = min(24, max_period)  # Default to daily or max if shorter
            
            period = st.slider(
                "Seasonal Period",
                min_value=min_period,
                max_value=max_period,
                value=default_period,
                step=4,
                help=f"24=Daily, 168=Weekly. Adjust based on your data patterns. Max period is {max_period}."
            )
        
        with col2:
            model_type = st.selectbox(
                "Decomposition Type",
                ["additive", "multiplicative"],
                help="Additive: Components are added together\nMultiplicative: Components are multiplied"
            )
        
        # Show decomposition
        with st.spinner("Analyzing time series components..."):
            fig = plot_decomposition(series, period, model_type, resample_rule)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        
        # Add explanatory text
        st.markdown("""
        ### Understanding the Results
        
        - **Original Data**: The raw time series with a smoothed line to show the general pattern
        - **Trend**: The long-term progression of the values
        - **Seasonal**: Regular patterns that repeat at fixed intervals
        - **Residual**: Random variations that can't be explained by trend or seasonality
        
        #### Tips for Interpretation
        - Look for clear patterns in the seasonal component
        - Check if the trend shows any long-term changes
        - Large residuals might indicate unusual events
        """)

if __name__ == "__main__":
    decomposition_page() 