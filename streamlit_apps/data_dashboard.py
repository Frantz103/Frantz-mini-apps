"""
Data Dashboard
A simple data visualization dashboard using Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# Page config
st.set_page_config(
    page_title="Data Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Data Visualization Dashboard")
st.markdown("---")


# Sidebar
with st.sidebar:
    st.header("Settings")
    chart_type = st.selectbox(
        "Select Chart Type",
        ["Line Chart", "Bar Chart", "Scatter Plot", "Area Chart", "Histogram"]
    )
    n_points = st.slider("Number of Data Points", 10, 500, 100)
    st.markdown("---")
    st.info("This is a demo dashboard. Upload your own data or connect to your data source!")


# Generate sample data
@st.cache_data
def generate_sample_data(n):
    """Generate sample time series data."""
    dates = pd.date_range(start='2024-01-01', periods=n, freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Value_A': np.cumsum(np.random.randn(n)) + 100,
        'Value_B': np.cumsum(np.random.randn(n)) + 50,
        'Category': np.random.choice(['Category 1', 'Category 2', 'Category 3'], n),
        'Random': np.random.rand(n) * 100
    })
    return data


df = generate_sample_data(n_points)


# Main content
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Average Value A",
        value=f"{df['Value_A'].mean():.2f}",
        delta=f"{df['Value_A'].iloc[-1] - df['Value_A'].iloc[0]:.2f}"
    )

with col2:
    st.metric(
        label="Average Value B",
        value=f"{df['Value_B'].mean():.2f}",
        delta=f"{df['Value_B'].iloc[-1] - df['Value_B'].iloc[0]:.2f}"
    )

with col3:
    st.metric(
        label="Max Value A",
        value=f"{df['Value_A'].max():.2f}",
    )

with col4:
    st.metric(
        label="Min Value A",
        value=f"{df['Value_A'].min():.2f}",
    )

st.markdown("---")

# Chart area
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader(f"{chart_type} Visualization")

    if chart_type == "Line Chart":
        fig = px.line(df, x='Date', y=['Value_A', 'Value_B'],
                     title='Time Series Data')
    elif chart_type == "Bar Chart":
        fig = px.bar(df.head(20), x='Date', y='Value_A',
                    title='Bar Chart')
    elif chart_type == "Scatter Plot":
        fig = px.scatter(df, x='Value_A', y='Value_B', color='Category',
                        title='Scatter Plot')
    elif chart_type == "Area Chart":
        fig = px.area(df, x='Date', y=['Value_A', 'Value_B'],
                     title='Area Chart')
    else:  # Histogram
        fig = px.histogram(df, x='Value_A', nbins=30,
                          title='Distribution of Value A')

    st.plotly_chart(fig, use_container_width=True)

with col_chart2:
    st.subheader("Category Distribution")
    category_counts = df['Category'].value_counts()
    fig_pie = px.pie(values=category_counts.values,
                     names=category_counts.index,
                     title='Category Distribution')
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# Data table
with st.expander("📋 View Raw Data"):
    st.dataframe(df, use_container_width=True)

# Download button
csv = df.to_csv(index=False)
st.download_button(
    label="📥 Download Data as CSV",
    data=csv,
    file_name="dashboard_data.csv",
    mime="text/csv",
)
