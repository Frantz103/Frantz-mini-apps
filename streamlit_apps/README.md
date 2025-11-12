# Streamlit Apps

This directory contains Streamlit applications for data analysis and visualization.

## Available Apps

1. **data_dashboard.py** - Interactive data visualization dashboard
2. **eda_tool.py** - Exploratory Data Analysis tool with file upload
3. **ml_trainer.py** - Train and evaluate ML models interactively

## Running an App

```bash
# Run a specific app
streamlit run streamlit_apps/data_dashboard.py

# Or
streamlit run streamlit_apps/eda_tool.py

# Or
streamlit run streamlit_apps/ml_trainer.py
```

The app will open in your default browser (usually on port 8501).

## Features

### Data Dashboard
- Multiple chart types (line, bar, scatter, area, histogram)
- Real-time data updates
- Interactive visualizations with Plotly
- Download data as CSV

### EDA Tool
- Upload your own CSV files
- Automatic statistical summary
- Missing data analysis
- Distribution analysis
- Correlation heatmaps
- Interactive filtering

### ML Trainer
- Train classification and regression models
- Multiple model types (Random Forest, Linear models, SVM)
- Feature importance visualization
- Model evaluation metrics
- Use sample data or upload your own

## Creating Your Own App

```python
import streamlit as st
import pandas as pd

st.title("My Streamlit App")

# Widgets
name = st.text_input("Enter your name")
st.write(f"Hello, {name}!")

# Display data
df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
st.dataframe(df)
```

Learn more at: https://docs.streamlit.io
