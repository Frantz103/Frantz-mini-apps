# Lightweight Python image for ML experiments
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY gradio_apps/ ./gradio_apps/
COPY streamlit_apps/ ./streamlit_apps/
COPY notebooks/ ./notebooks/
COPY utils/ ./utils/
COPY data/ ./data/
COPY models/ ./models/

# Create directories if they don't exist
RUN mkdir -p data models logs

# Expose ports
# 7860-7862: Gradio apps
# 8501: Streamlit
# 8888: Jupyter
EXPOSE 7860 7861 7862 8501 8888

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Default command (can be overridden)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
