#!/bin/bash
# Script to run Gradio apps

echo "Available Gradio Apps:"
echo "1. Image Classifier (port 7860)"
echo "2. Text Analyzer (port 7861)"
echo "3. Simple ML Demo (port 7862)"
echo ""
echo "Enter your choice (1-3):"
read choice

case $choice in
    1)
        echo "Starting Image Classifier..."
        python gradio_apps/image_classifier.py
        ;;
    2)
        echo "Starting Text Analyzer..."
        python gradio_apps/text_analyzer.py
        ;;
    3)
        echo "Starting Simple ML Demo..."
        python gradio_apps/simple_ml_demo.py
        ;;
    *)
        echo "Invalid choice. Please run again and select 1-3."
        exit 1
        ;;
esac
