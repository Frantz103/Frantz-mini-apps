"""
Exploratory Data Analysis Tool
Upload and analyze your datasets with this Streamlit app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from io import StringIO


# Page config
st.set_page_config(
    page_title="EDA Tool",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Exploratory Data Analysis Tool")
st.markdown("Upload your dataset to perform quick exploratory analysis.")
st.markdown("---")


# File uploader
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload a CSV file to analyze"
)

# Sample data option
if st.checkbox("Use sample data instead"):
    uploaded_file = None
    # Generate sample data
    np.random.seed(42)
    sample_df = pd.DataFrame({
        'age': np.random.randint(18, 80, 200),
        'income': np.random.randint(20000, 150000, 200),
        'score': np.random.randint(0, 100, 200),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 200),
        'is_active': np.random.choice([True, False], 200)
    })
    df = sample_df
else:
    df = None


# Load data
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        st.error(f"Error loading file: {e}")


# Main analysis
if df is not None:
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Distributions", "🔗 Correlations", "📋 Data"])

    with tab1:
        st.header("Dataset Overview")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())

        st.subheader("Data Types")
        st.dataframe(pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values
        }), use_container_width=True)

        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

        # Missing data visualization
        if df.isnull().sum().sum() > 0:
            st.subheader("Missing Data Pattern")
            missing_df = df.isnull().sum().reset_index()
            missing_df.columns = ['Column', 'Missing Count']
            missing_df = missing_df[missing_df['Missing Count'] > 0]

            fig = px.bar(missing_df, x='Column', y='Missing Count',
                        title='Missing Values by Column')
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Distribution Analysis")

        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            selected_col = st.selectbox("Select column to visualize", numeric_cols)

            col_viz1, col_viz2 = st.columns(2)

            with col_viz1:
                st.subheader("Histogram")
                fig_hist = px.histogram(df, x=selected_col, nbins=30,
                                       title=f'Distribution of {selected_col}')
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_viz2:
                st.subheader("Box Plot")
                fig_box = px.box(df, y=selected_col,
                                title=f'Box Plot of {selected_col}')
                st.plotly_chart(fig_box, use_container_width=True)

            # Categorical breakdown if available
            categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
            if categorical_cols:
                st.subheader("Breakdown by Category")
                cat_col = st.selectbox("Select categorical column", categorical_cols)
                fig_violin = px.violin(df, y=selected_col, x=cat_col, box=True,
                                      title=f'{selected_col} by {cat_col}')
                st.plotly_chart(fig_violin, use_container_width=True)
        else:
            st.info("No numeric columns found in the dataset.")

    with tab3:
        st.header("Correlation Analysis")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) >= 2:
            # Correlation matrix
            corr_matrix = df[numeric_cols].corr()

            st.subheader("Correlation Heatmap")
            fig_corr = px.imshow(corr_matrix,
                                text_auto=True,
                                aspect="auto",
                                title="Correlation Matrix",
                                color_continuous_scale='RdBu_r',
                                zmin=-1, zmax=1)
            st.plotly_chart(fig_corr, use_container_width=True)

            # Scatter plot
            st.subheader("Scatter Plot")
            col_scatter1, col_scatter2 = st.columns(2)
            with col_scatter1:
                x_axis = st.selectbox("X-axis", numeric_cols)
            with col_scatter2:
                y_axis = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))

            fig_scatter = px.scatter(df, x=x_axis, y=y_axis,
                                    title=f'{y_axis} vs {x_axis}',
                                    trendline="ols")
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis.")

    with tab4:
        st.header("Raw Data")

        # Filtering options
        st.subheader("Filter Data")
        col_filter1, col_filter2 = st.columns(2)

        with col_filter1:
            rows_to_show = st.slider("Number of rows to display", 5, len(df), min(100, len(df)))

        with col_filter2:
            columns_to_show = st.multiselect(
                "Select columns to display",
                df.columns.tolist(),
                default=df.columns.tolist()[:5]
            )

        if columns_to_show:
            st.dataframe(df[columns_to_show].head(rows_to_show), use_container_width=True)
        else:
            st.dataframe(df.head(rows_to_show), use_container_width=True)

        # Download filtered data
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Dataset",
            data=csv,
            file_name="analyzed_data.csv",
            mime="text/csv",
        )

else:
    st.info("👆 Please upload a CSV file or check 'Use sample data' to get started!")
