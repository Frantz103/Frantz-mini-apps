#!/bin/bash
# Script to run Streamlit apps

echo "Available Streamlit Apps:"
echo "1. Data Dashboard"
echo "2. EDA Tool"
echo "3. ML Trainer"
echo ""
echo "Enter your choice (1-3):"
read choice

case $choice in
    1)
        echo "Starting Data Dashboard..."
        streamlit run streamlit_apps/data_dashboard.py
        ;;
    2)
        echo "Starting EDA Tool..."
        streamlit run streamlit_apps/eda_tool.py
        ;;
    3)
        echo "Starting ML Trainer..."
        streamlit run streamlit_apps/ml_trainer.py
        ;;
    *)
        echo "Invalid choice. Please run again and select 1-3."
        exit 1
        ;;
esac
