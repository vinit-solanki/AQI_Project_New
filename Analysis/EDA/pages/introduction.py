import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (preprocess_timeseries, load_data, filter_time_range, 
                  TIME_RANGES, RESAMPLE_RULES, POLLUTANTS, THEME)

def show_introduction():
    st.title("Time Series Analysis Dashboard")
    
    st.markdown("""
    ### Understanding Air Quality Time Series Data
    
    This dashboard provides comprehensive analysis tools for air quality data, helping you:
    - Visualize trends and patterns
    - Analyze seasonal variations
    - Test for stationarity
    - Examine correlations
    
    Use the navigation menu on the left to explore different analyses.
    """)
    
    # Create tabs for different components
    tabs = st.tabs(["Components", "Analysis Tools", "Data Handling"])
    
    with tabs[0]:
        st.markdown("""
        #### Key Components in Air Quality Data
        
        1. **Trend**
           - Long-term changes in pollution levels
           - Overall increase or decrease in concentrations
           - Impact of environmental policies
        
        2. **Patterns**
           - Daily variations (rush hours, industrial activity)
           - Weekly cycles (workday vs weekend)
           - Monthly/Seasonal changes (weather impact)
        
        3. **Random Variations**
           - Unexpected events
           - Measurement uncertainties
           - Short-term fluctuations
        """)
        
    with tabs[1]:
        st.markdown("""
        #### Available Analysis Tools
        
        1. **Stationarity Analysis**
           - Tests for data stability
           - Trend identification
           - Transformation suggestions
        
        2. **Decomposition**
           - Separate trend, seasonal, and random components
           - Identify underlying patterns
           - Analyze each component individually
        
        3. **Correlation Analysis**
           - ACF/PACF plots
           - Pattern identification
           - Model suggestions
        """)
        
    with tabs[2]:
        st.markdown("""
        #### Data Processing Features
        
        - **Time Range Selection**: Analyze specific periods
        - **Resampling**: Adjust data frequency
        - **Outlier Handling**: Remove extreme values
        - **Missing Value Treatment**: Interpolation methods
        """)

def show_example_visualization():
    st.header("Interactive Data Visualization")
    
    # Add custom CSS for dark theme
    st.markdown("""
        <style>
        .stSelectbox {
            background-color: #1a1a1a;
            color: white;
            border-radius: 5px;
            padding: 5px;
        }
        .stSlider {
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .css-145kmo2 {
            color: white;
        }
        .css-1d391kg {
            background-color: #1a1a1a;
        }
        </style>
    """, unsafe_allow_html=True)
    
    df = load_data()
    
    if df is not None:
        # Create three columns for controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pollutant = st.selectbox(
                "Select Pollutant",
                POLLUTANTS,
                help="Choose the air quality parameter to visualize"
            )
        
        with col2:
            time_range = st.selectbox(
                "Select Time Range",
                list(TIME_RANGES.keys()),
                format_func=lambda x: TIME_RANGES[x],
                index=1,  # Default to 1 month
                help="Choose the time period to display"
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
        
        # Preprocess and plot the data
        series_clean = preprocess_timeseries(series, resample_rule)
        
        fig = go.Figure()
        
        # Add raw data with low opacity
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series,
            mode='lines',
            name='Raw Data',
            line=dict(color='gray', width=1),
            opacity=0.3
        ))
        
        # Add cleaned and resampled data
        fig.add_trace(go.Scatter(
            x=series_clean.index,
            y=series_clean,
            mode='lines',
            name='Processed Data',
            line=dict(color=THEME['primary_color'], width=2)
        ))
        
        # Update layout with dark theme
        fig.update_layout(
            title=f"{pollutant} Concentration Over Time",
            xaxis_title="Date",
            yaxis_title="Concentration",
            template=THEME['plot_template'],
            plot_bgcolor=THEME['plot_bgcolor'],
            paper_bgcolor=THEME['paper_bgcolor'],
            height=500,
            showlegend=True,
            font=dict(
                family=THEME['font_family'],
                color=THEME['text_color']
            ),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor=THEME['grid_color'],
                color=THEME['axis_color']
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor=THEME['grid_color'],
                color=THEME['axis_color']
            ),
            legend=dict(
                font=dict(color=THEME['legend_font_color'])
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add data statistics with dark theme styling
        st.markdown("""
        <style>
        .dataframe {
            background-color: #1a1a1a;
            color: white;
        }
        .dataframe th {
            background-color: #333333;
            color: white;
        }
        .dataframe td {
            background-color: #1a1a1a;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.subheader("Data Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Summary Statistics")
            stats = series_clean.describe()
            st.write(pd.DataFrame({
                'Statistic': stats.index,
                'Value': stats.values.round(2)
            }).set_index('Statistic'))
        
        with col2:
            st.markdown("#### Data Quality")
            missing = series.isnull().sum()
            total = len(series)
            st.write(f"- Total Observations: {total:,}")
            st.write(f"- Missing Values: {missing:,} ({(missing/total*100):.1f}%)")
            st.write(f"- Time Range: {series.index.min().date()} to {series.index.max().date()}")

def introduction_page():
    show_introduction()
    show_example_visualization()

if __name__ == "__main__":
    introduction_page() 