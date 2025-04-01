import streamlit as st
from pages.introduction import introduction_page
from pages.stationarity import stationarity_page
from pages.decomposition import decomposition_page
from pages.correlation import correlation_page

def main():
    st.set_page_config(
        page_title="Time Series Analysis Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    pages = {
        "Introduction": introduction_page,
        "Stationarity Analysis": stationarity_page,
        "Time Series Decomposition": decomposition_page,
        "Correlation Analysis": correlation_page
    }
    
    # Add description for each page
    page_descriptions = {
        "Introduction": "Learn about time series data and its components",
        "Stationarity Analysis": "Test and transform data for stationarity",
        "Time Series Decomposition": "Break down time series into components",
        "Correlation Analysis": "Analyze ACF and PACF patterns"
    }
    
    # Show page descriptions in sidebar
    st.sidebar.markdown("### Page Descriptions")
    for page, desc in page_descriptions.items():
        st.sidebar.markdown(f"**{page}**: {desc}")
    
    st.sidebar.markdown("---")
    
    # Page selection
    selected_page = st.sidebar.radio("Select a page", list(pages.keys()))
    
    # Display selected page
    pages[selected_page]()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This dashboard provides comprehensive time series analysis tools for air quality data.
    It includes:
    - Basic time series concepts
    - Stationarity testing and transformation
    - Time series decomposition
    - ACF/PACF analysis
    """)

if __name__ == "__main__":
    main() 