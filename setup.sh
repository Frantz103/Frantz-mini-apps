#!/bin/bash
# Setup script for Frantz Mini Apps ML Environment

set -e

echo "========================================="
echo "Frantz Mini Apps - ML Environment Setup"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if Python 3.9+ is installed
if ! python3 -c 'import sys; assert sys.version_info >= (3, 9)' 2>/dev/null; then
    echo "Error: Python 3.9 or higher is required"
    echo "Please install Python 3.9+ and try again"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo "Virtual environment created successfully!"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x run_gradio.sh
chmod +x run_streamlit.sh
chmod +x setup.sh

# Create necessary directories
echo ""
echo "Creating necessary directories..."
mkdir -p data models logs

echo ""
echo "========================================="
echo "Setup completed successfully!"
echo "========================================="
echo ""
echo "To get started:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run Jupyter Lab:"
echo "     jupyter lab"
echo ""
echo "  3. Or run a Gradio app:"
echo "     ./run_gradio.sh"
echo ""
echo "  4. Or run a Streamlit app:"
echo "     ./run_streamlit.sh"
echo ""
echo "  5. Or use Docker:"
echo "     docker-compose up"
echo ""
echo "Happy experimenting!"
