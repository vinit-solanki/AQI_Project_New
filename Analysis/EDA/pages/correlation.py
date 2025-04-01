import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    preprocess_timeseries, load_data, filter_time_range, 
    TIME_RANGES, RESAMPLE_RULES, POLLUTANTS, THEME
)

# Custom CSS for professional UI
st.markdown("""
    <style>
    /* Main container styling */
    .stApp {
        background-color: #1a1a1a;
        padding: 20px;
    }
    /* Title styling */
    h1 {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 20px;
    }
    /* Subheader styling */
    h2 {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    /* Widget styling */
    .stSelectbox, .stSlider, .stButton {
        background-color: #2d2d2d;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    /* Plot container styling */
    .stPlotlyChart {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        background-color: #2d2d2d;
    }
    /* Info box styling */
    .info-box {
        background-color: #2d2d2d;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        color: #ffffff;
    }
    /* Interpretation box styling */
    .interpretation-box {
        background-color: #363636;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        color: #ffffff;
    }
    /* Pattern analysis box styling */
    .pattern-analysis {
        background-color: #2d2d2d;
        color: #ffffff;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    /* Table styling */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
        color: #ffffff;
    }
    th, td {
        padding: 10px;
        text-align: left;
        border-bottom: 1px solid #404040;
    }
    th {
        background-color: #363636;
        color: #ffffff;
    }
    td {
        color: #ffffff;
    }
    
    /* Additional dark theme elements */
    p, li {
        color: #ffffff;
    }
    .streamlit-expanderHeader {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
    }
    .stMarkdown {
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

def plot_acf_pacf(series, nlags=40, alpha=0.05, resample_rule='1H'):
    """Plot ACF and PACF with enhanced styling"""
    series_clean = preprocess_timeseries(series, resample_rule)
    acf_values = acf(series_clean, nlags=nlags, alpha=alpha)
    pacf_values = pacf(series_clean, nlags=nlags, alpha=alpha)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Autocorrelation Function (ACF)", "Partial Autocorrelation Function (PACF)"),
        vertical_spacing=0.15
    )

    # ACF Plot with enhanced confidence intervals
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(acf_values[0])),
            y=acf_values[0],
            mode='lines+markers',
            name='ACF',
            line=dict(color=THEME['primary_color'], width=2),
            marker=dict(size=6, color=THEME['primary_color'])
        ),
        row=1, col=1
    )
    
    # Add horizontal reference lines at 0
    fig.add_hline(
        y=0,
        line=dict(color='rgba(255, 255, 255, 0.5)', width=1, dash='dot'),
        row=1, col=1
    )
    
    # Confidence intervals for ACF
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(acf_values[0])),
            y=acf_values[1][:, 0],
            mode='lines',
            line=dict(dash='dash', color='rgba(255, 255, 255, 0.7)', width=1),
            name='Confidence Interval',
            showlegend=True
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(acf_values[0])),
            y=acf_values[1][:, 1],
            mode='lines',
            line=dict(dash='dash', color='rgba(255, 255, 255, 0.7)', width=1),
            fill='tonexty',
            fillcolor='rgba(255, 255, 255, 0.1)',
            name='Confidence Interval',
            showlegend=False
        ),
        row=1, col=1
    )

    # PACF Plot with enhanced confidence intervals
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(pacf_values[0])),
            y=pacf_values[0],
            mode='lines+markers',
            name='PACF',
            line=dict(color=THEME['secondary_color'], width=2),
            marker=dict(size=6, color=THEME['secondary_color'])
        ),
        row=2, col=1
    )
    
    # Add horizontal reference lines at 0
    fig.add_hline(
        y=0,
        line=dict(color='rgba(255, 255, 255, 0.5)', width=1, dash='dot'),
        row=2, col=1
    )
    
    # Confidence intervals for PACF
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(pacf_values[0])),
            y=pacf_values[1][:, 0],
            mode='lines',
            line=dict(dash='dash', color='rgba(255, 255, 255, 0.7)', width=1),
            name='Confidence Interval',
            showlegend=False
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(pacf_values[0])),
            y=pacf_values[1][:, 1],
            mode='lines',
            line=dict(dash='dash', color='rgba(255, 255, 255, 0.7)', width=1),
            fill='tonexty',
            fillcolor='rgba(255, 255, 255, 0.1)',
            name='Confidence Interval',
            showlegend=False
        ),
        row=2, col=1
    )

    # Update layout with dark theme
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Correlation Analysis",
        template='plotly_dark',
        plot_bgcolor='rgba(45, 45, 45, 0.8)',
        paper_bgcolor='rgba(45, 45, 45, 0.8)',
        font=dict(color='#ffffff'),
        title=dict(
            font=dict(size=24, color='#ffffff'),
            x=0.5,
            y=0.95
        )
    )
    
    # Update axes with dark theme
    for i in range(1, 3):
        fig.update_xaxes(
            title_text="Lag",
            row=i, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            title_font=dict(size=14, color='#ffffff')
        )
        fig.update_yaxes(
            title_text="ACF" if i == 1 else "PACF",
            row=i, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            title_font=dict(size=14, color='#ffffff')
        )

    return fig

def plot_simple_acf_pacf(series, nlags=40, alpha=0.05, resample_rule='1H'):
    """Plot ACF and PACF with enhanced dark theme styling"""
    series_clean = preprocess_timeseries(series, resample_rule)
    acf_values = acf(series_clean, nlags=nlags, alpha=alpha)
    pacf_values = pacf(series_clean, nlags=nlags, alpha=alpha)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "<b>Autocorrelation Function (ACF)</b>",
            "<b>Partial Autocorrelation Function (PACF)</b>"
        ),
        vertical_spacing=0.2
    )

    # Custom color scheme
    bar_color = '#00bfff'  # Deep sky blue
    ci_color = '#ff1493'   # Deep pink for better visibility
    grid_color = 'rgba(255, 255, 255, 0.1)'
    zero_line_color = 'rgba(255, 255, 255, 0.3)'
    
    # Calculate CI value for annotation
    ci_value = acf_values[1][:, 1][0]
    ci_text = f"95% CI: ±{ci_value:.3f}"

    # ACF Plot with enhanced styling
    fig.add_trace(
        go.Bar(
            x=np.arange(len(acf_values[0])),
            y=acf_values[0],
            name='ACF',
            marker=dict(
                color=bar_color,
                opacity=0.7,
                line=dict(color='#ffffff', width=1)
            ),
            width=0.6
        ),
        row=1, col=1
    )
    
    # Add confidence intervals for ACF with enhanced visibility
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(acf_values[0])),
            y=[ci_value]*len(acf_values[0]),
            mode='lines',
            line=dict(color=ci_color, width=3, dash='dash'),
            name='95% Confidence Level',
            hovertemplate='Upper CI: %{y:.3f}<extra></extra>',
            showlegend=True
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(acf_values[0])),
            y=[-ci_value]*len(acf_values[0]),
            mode='lines',
            line=dict(color=ci_color, width=3, dash='dash'),
            hovertemplate='Lower CI: %{y:.3f}<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )

    # Add CI annotation for ACF
    fig.add_annotation(
        x=nlags-5, y=ci_value,
        text=ci_text,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor=ci_color,
        font=dict(color=ci_color, size=12),
        bgcolor='rgba(0,0,0,0.7)',
        bordercolor=ci_color,
        borderwidth=2,
        borderpad=4,
        row=1, col=1
    )

    # PACF Plot with enhanced styling
    fig.add_trace(
        go.Bar(
            x=np.arange(len(pacf_values[0])),
            y=pacf_values[0],
            name='PACF',
            marker=dict(
                color=bar_color,
                opacity=0.7,
                line=dict(color='#ffffff', width=1)
            ),
            width=0.6,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add confidence intervals for PACF with enhanced visibility
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(pacf_values[0])),
            y=[ci_value]*len(pacf_values[0]),
            mode='lines',
            line=dict(color=ci_color, width=3, dash='dash'),
            hovertemplate='Upper CI: %{y:.3f}<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(pacf_values[0])),
            y=[-ci_value]*len(pacf_values[0]),
            mode='lines',
            line=dict(color=ci_color, width=3, dash='dash'),
            hovertemplate='Lower CI: %{y:.3f}<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )

    # Add CI annotation for PACF
    fig.add_annotation(
        x=nlags-5, y=ci_value,
        text=ci_text,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor=ci_color,
        font=dict(color=ci_color, size=12),
        bgcolor='rgba(0,0,0,0.7)',
        bordercolor=ci_color,
        borderwidth=2,
        borderpad=4,
        row=2, col=1
    )

    # Update layout with dark theme
    fig.update_layout(
        height=800,
        showlegend=True,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(
            color='white',
            size=14,
            family='Arial'
        ),
        title_font=dict(size=16),
        margin=dict(t=80, b=40, l=60, r=40),
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1
        )
    )

    # Update axes with enhanced styling
    for i in range(1, 3):
        fig.update_xaxes(
            title_text="Lags",
            title_font=dict(size=14),
            row=i, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor=grid_color,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor=zero_line_color,
            tickfont=dict(size=12)
        )
        fig.update_yaxes(
            title_text="Correlation" if i == 1 else "Partial Correlation",
            title_font=dict(size=14),
            row=i, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor=grid_color,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor=zero_line_color,
            range=[-1.1, 1.1],  # Slightly expanded range
            tickfont=dict(size=12)
        )

    return fig

def plot_statsmodels_acf_pacf(data_path, column='PM2.5', lags=30):
    """
    Plot ACF and PACF using statsmodels with enhanced styling for both original and differenced series
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import seaborn as sns
    
    # Set style for better visibility
    plt.style.use('dark_background')
    sns.set_style("darkgrid", {"axes.facecolor": ".1"})
    
    # Read and prepare data
    df = pd.read_csv(data_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    
    # Get the series and its difference
    series = df[column].dropna()
    series_diff = series.diff().dropna()
    
    # Create figure with dark theme
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor('#1a1a1a')
    
    # Plot ACF of original series
    plot_acf(series, lags=lags, ax=axes[0,0], alpha=0.05, title=None)
    axes[0,0].set_title(f'ACF of {column}', color='white', fontsize=12, pad=10)
    
    # Plot PACF of original series
    plot_pacf(series, lags=lags, ax=axes[0,1], alpha=0.05, title=None)
    axes[0,1].set_title(f'PACF of {column}', color='white', fontsize=12, pad=10)
    
    # Plot ACF of differenced series
    plot_acf(series_diff, lags=lags, ax=axes[1,0], alpha=0.05, title=None)
    axes[1,0].set_title(f'ACF of Differenced {column}', color='white', fontsize=12, pad=10)
    
    # Plot PACF of differenced series
    plot_pacf(series_diff, lags=lags, ax=axes[1,1], alpha=0.05, title=None)
    axes[1,1].set_title(f'PACF of Differenced {column}', color='white', fontsize=12, pad=10)
    
    # Enhance styling for all subplots
    for ax in axes.flat:
        # Set colors for better visibility
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        
        # Make confidence intervals more visible
        if len(ax.lines) > 0:
            ax.lines[0].set_color('#00bfff')  # Main bars
            if len(ax.lines) > 1:
                ax.lines[1].set_color('#ff1493')  # Upper CI
                ax.lines[2].set_color('#ff1493')  # Lower CI
                ax.lines[1].set_linestyle('--')
                ax.lines[2].set_linestyle('--')
                ax.lines[1].set_linewidth(1.5)
                ax.lines[2].set_linewidth(1.5)
    
    # Add overall title and adjust layout
    fig.suptitle(f'ACF and PACF Analysis for {column}', color='white', fontsize=14, y=0.95)
    plt.tight_layout()
    return fig

def analyze_correlations(data_path='datasets/KurlaMumbaiMPCB.csv'):
    """
    Analyze and display ACF/PACF plots for the AQI data
    """
    import streamlit as st
    
    st.subheader("Time Series Correlation Analysis")
    st.write("Analyzing autocorrelation and partial autocorrelation functions for PM2.5")
    
    # Create and display the plots
    fig = plot_statsmodels_acf_pacf(data_path)
    st.pyplot(fig)
    
    st.markdown("""
    ### Interpretation Guide:
    - **ACF (Autocorrelation Function)**: Shows correlation between observations at different time lags
    - **PACF (Partial Autocorrelation Function)**: Shows direct correlation between observations
    - **Differenced Series**: Helps analyze trends after removing time-dependent patterns
    - **Blue Lines**: Correlation values at each lag
    - **Pink Dashed Lines**: 95% Confidence Intervals
    - **Significant Correlations**: Bars extending beyond the confidence intervals
    """)

def display_significant_lags(title, lags, confidence_interval):
    """Display significant lags with clearer visualization"""
    status_color = '#2ecc71' if len(lags) > 0 else '#e74c3c'
    
    st.markdown(f"""
    <div class='info-box' style='background: linear-gradient(135deg, #2d2d2d 0%, #363636 100%);'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;'>
            <h3 style='color: #ffffff; margin: 0; font-size: 1.4em;'>{title}</h3>
            <div style='background-color: {status_color}; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold;'>
                {len(lags)} Patterns Found
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if len(lags) > 0:
        # Group lags into meaningful categories
        patterns = {
            'hourly': [lag for lag in lags if lag <= 12],
            'daily': [lag for lag in lags if 12 < lag <= 24],
            'multi_day': [lag for lag in lags if lag > 24]
        }
        
        # Display pattern groups with icons
        st.markdown("""
        <div style='background-color: #404040; padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h4 style='color: white; margin-bottom: 20px;'>🔍 Found Patterns:</h4>
        """, unsafe_allow_html=True)
        
        # Hourly patterns
        if patterns['hourly']:
            st.markdown("""
            <div style='background: linear-gradient(90deg, #3498db20, #3498db40); padding: 15px; border-radius: 10px; margin: 10px 0;'>
                <h5 style='color: #3498db; margin: 0;'>⏰ Hourly Patterns</h5>
                <div style='display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;'>
            """, unsafe_allow_html=True)
            
            for lag in patterns['hourly']:
                st.markdown(f"""
                    <div style='background: #3498db30; padding: 8px 15px; border-radius: 20px;'>
                        <span style='color: white;'>Every {lag} hours</span>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Daily patterns
        if patterns['daily']:
            st.markdown("""
            <div style='background: linear-gradient(90deg, #2ecc7120, #2ecc7140); padding: 15px; border-radius: 10px; margin: 10px 0;'>
                <h5 style='color: #2ecc71; margin: 0;'>📅 Daily Patterns</h5>
                <div style='display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;'>
            """, unsafe_allow_html=True)
            
            for lag in patterns['daily']:
                st.markdown(f"""
                    <div style='background: #2ecc7130; padding: 8px 15px; border-radius: 20px;'>
                        <span style='color: white;'>{lag} hours (≈ 1 day)</span>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Multi-day patterns
        if patterns['multi_day']:
            st.markdown("""
            <div style='background: linear-gradient(90deg, #e67e2220, #e67e2240); padding: 15px; border-radius: 10px; margin: 10px 0;'>
                <h5 style='color: #e67e22; margin: 0;'>📊 Long-term Patterns</h5>
                <div style='display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;'>
            """, unsafe_allow_html=True)
            
            for lag in patterns['multi_day']:
                days = lag // 24
                hours = lag % 24
                time_str = f"{days}d {hours}h" if hours else f"{days} days"
                st.markdown(f"""
                    <div style='background: #e67e2230; padding: 8px 15px; border-radius: 20px;'>
                        <span style='color: white;'>Every {time_str}</span>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Simple interpretation
        st.markdown(f"""
        <div style='background-color: #2d2d2d; padding: 15px; border-radius: 10px; margin-top: 20px;'>
            <h4 style='color: #3498db; margin: 0 0 10px 0;'>💡 What This Means:</h4>
            <p style='color: white; margin: 0;'>
                Your data shows repeating patterns:
                {f"<br>• Every few hours ({len(patterns['hourly'])} patterns)" if patterns['hourly'] else ""}
                {f"<br>• Daily cycles ({len(patterns['daily'])} patterns)" if patterns['daily'] else ""}
                {f"<br>• Longer-term cycles ({len(patterns['multi_day'])} patterns)" if patterns['multi_day'] else ""}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # No patterns found
        st.markdown("""
        <div style='background-color: #404040; padding: 20px; border-radius: 10px; text-align: center;'>
            <div style='font-size: 2em; margin-bottom: 10px;'>🔍</div>
            <p style='color: #e74c3c; margin: 0; font-size: 1.1em;'>No Clear Patterns Found</p>
            <p style='color: #ffffff; margin: 10px 0 0 0;'>The data appears to be more random in nature.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def correlation_page():
    st.title("Time Series Correlation Analysis")
    
    # Main info box with better spacing and alignment
    st.markdown("""
    <div class='info-box' style='margin-bottom: 30px;'>
        <h3 style='margin-bottom: 15px; color: #3498db;'>Understanding Time Series Correlations</h3>
        <p style='margin-bottom: 15px;'>This analysis helps identify patterns and dependencies in your air quality data through two key measures:</p>
        <ul style='list-style-type: none; padding-left: 0;'>
            <li style='margin-bottom: 10px; padding-left: 25px; position: relative;'>
                <span style='position: absolute; left: 0; color: #3498db;'>📈</span>
                <strong>Autocorrelation Function (ACF):</strong> Shows overall correlation patterns, including seasonality and cycles.
            </li>
            <li style='margin-bottom: 10px; padding-left: 25px; position: relative;'>
                <span style='position: absolute; left: 0; color: #3498db;'>📊</span>
                <strong>Partial Autocorrelation Function (PACF):</strong> Shows direct correlation patterns, helping identify autoregressive terms.
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    if df is not None:
        # Better organized control panel
        st.markdown("<div style='background-color: #2d2d2d; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            pollutant = st.selectbox("Select Pollutant", POLLUTANTS, help="Choose the air quality parameter to analyze")
        with col2:
            time_range = st.selectbox("Select Time Range", list(TIME_RANGES.keys()), format_func=lambda x: TIME_RANGES[x], index=1)
        with col3:
            resample_rule = st.selectbox("Select Data Frequency", list(RESAMPLE_RULES.keys()), format_func=lambda x: RESAMPLE_RULES[x])
        st.markdown("</div>", unsafe_allow_html=True)

        # Analysis settings with better organization
        st.markdown("""
        <div style='background-color: #2d2d2d; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
            <h3 style='margin-bottom: 20px; color: #3498db;'>Analysis Settings</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            nlags = st.slider("Number of Lags", min_value=10, max_value=100, value=40, 
                             help="Number of time lags to analyze")
        with col2:
            confidence = st.slider("Confidence Level", min_value=90, max_value=99, value=95, 
                                 help="Statistical confidence level for significance bounds")
        st.markdown("</div>", unsafe_allow_html=True)

        alpha = 1 - confidence / 100

        with st.spinner("Analyzing correlation patterns..."):
            series_clean = preprocess_timeseries(df[pollutant], resample_rule)
            acf_values = acf(series_clean, nlags=nlags, alpha=alpha)
            pacf_values = pacf(series_clean, nlags=nlags, alpha=alpha)
            fig = plot_acf_pacf(df[pollutant], nlags, alpha, resample_rule)
            st.plotly_chart(fig, use_container_width=True)

        # Add the simple ACF/PACF plots
        st.subheader("Simple ACF/PACF Analysis")
        simple_fig = plot_simple_acf_pacf(df[pollutant], nlags=15, alpha=alpha, resample_rule=resample_rule)
        st.plotly_chart(simple_fig, use_container_width=True)

        st.subheader("Significant Correlations")
        col1, col2 = st.columns(2)
        with col1:
            significant_acf = np.where(np.abs(acf_values[0]) > acf_values[1][:, 1])[0]
            display_significant_lags("ACF Significant Lags", significant_acf, alpha)
        with col2:
            significant_pacf = np.where(np.abs(pacf_values[0]) > pacf_values[1][:, 1])[0]
            display_significant_lags("PACF Significant Lags", significant_pacf, alpha)

        st.markdown("""
        <div class='interpretation-box'>
            <h3>Detailed Interpretation</h3>
            <p><strong>ACF Analysis:</strong></p>
            <ul>
                <li>Shows the overall correlation between current and past values</li>
                <li>Helps identify seasonality and long-term patterns</li>
                <li>Significant lags indicate repeating patterns at those intervals</li>
            </ul>
            <p><strong>PACF Analysis:</strong></p>
            <ul>
                <li>Shows direct correlation between current and past values</li>
                <li>Removes indirect relationships through other time periods</li>
                <li>Helps determine the order of autoregressive models</li>
            </ul>
            <p><strong>Model Implications:</strong></p>
            <ul>
                <li>Significant PACF lags suggest AR terms in the model</li>
                <li>Significant ACF lags suggest MA terms in the model</li>
                <li>Patterns at specific intervals suggest seasonal components</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    correlation_page()