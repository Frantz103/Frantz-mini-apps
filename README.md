# Frantz Mini Apps - ML Experimentation Environment

A lightweight machine learning experimentation environment featuring **Gradio**, **Streamlit**, and **Jupyter Notebook** - optimized for small machines and quick prototyping.

## Features

- 🚀 **Quick Setup** - Get started in minutes with automated setup script
- 🎨 **Gradio Apps** - Interactive ML demos and interfaces
- 📊 **Streamlit Dashboards** - Data visualization and analysis tools
- 📓 **Jupyter Notebooks** - Exploratory data analysis and experiments
- 🐳 **Docker Support** - Containerized deployment for easy portability
- 🛠️ **Utility Functions** - Reusable modules for data, models, and visualizations
- 💻 **Lightweight** - Optimized for small machines and limited resources

## Project Structure

```
Frantz-mini-apps/
├── gradio_apps/           # Gradio demo applications
│   ├── image_classifier.py
│   ├── text_analyzer.py
│   └── simple_ml_demo.py
├── streamlit_apps/        # Streamlit data apps
│   ├── data_dashboard.py
│   ├── eda_tool.py
│   └── ml_trainer.py
├── notebooks/             # Jupyter notebooks
│   └── examples/
│       ├── 01_data_analysis.ipynb
│       ├── 02_ml_experiment.ipynb
│       └── 03_visualization_gallery.ipynb
├── utils/                 # Utility functions
│   ├── data_utils.py
│   ├── model_utils.py
│   └── viz_utils.py
├── data/                  # Data directory
├── models/                # Saved models
├── logs/                  # Experiment logs
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Project configuration
└── docker-compose.yml    # Docker configuration
```

## Quick Start

### Option 1: Local Setup (Recommended for Development)

#### Prerequisites
- Python 3.9 or higher
- pip

#### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Frantz-mini-apps
   ```

2. **Run the setup script**
   ```bash
   ./setup.sh
   ```

3. **Activate the virtual environment**
   ```bash
   source venv/bin/activate
   ```

4. **You're ready to go!**

### Option 2: Docker Setup (Recommended for Deployment)

#### Prerequisites
- Docker
- Docker Compose

#### Usage

1. **Build and run all services**
   ```bash
   docker-compose up
   ```

2. **Access the applications**
   - Jupyter Lab: http://localhost:8888
   - Gradio Image Classifier: http://localhost:7860
   - Gradio Text Analyzer: http://localhost:7861
   - Streamlit Dashboard: http://localhost:8501

3. **Run specific service**
   ```bash
   docker-compose up jupyter
   # or
   docker-compose up streamlit-dashboard
   ```

## Usage Guide

### Running Gradio Apps

Gradio provides quick, interactive interfaces for ML models.

```bash
# Interactive menu
./run_gradio.sh

# Or run directly
python gradio_apps/image_classifier.py
python gradio_apps/text_analyzer.py
python gradio_apps/simple_ml_demo.py
```

**Available Apps:**
- **Image Classifier** (port 7860) - Upload images for classification
- **Text Analyzer** (port 7861) - Analyze text statistics and word frequency
- **Simple ML Demo** (port 7862) - Train and visualize a simple ML model

### Running Streamlit Apps

Streamlit apps provide interactive data visualization and analysis.

```bash
# Interactive menu
./run_streamlit.sh

# Or run directly
streamlit run streamlit_apps/data_dashboard.py
streamlit run streamlit_apps/eda_tool.py
streamlit run streamlit_apps/ml_trainer.py
```

**Available Apps:**
- **Data Dashboard** - Interactive charts and visualizations
- **EDA Tool** - Upload and analyze CSV files
- **ML Trainer** - Train and compare ML models

### Running Jupyter Notebooks

Jupyter provides an interactive environment for exploratory analysis.

```bash
# Start Jupyter Lab (modern interface)
jupyter lab

# Or Jupyter Notebook (classic interface)
jupyter notebook

# Start from notebooks directory
jupyter lab notebooks/
```

**Example Notebooks:**
- `01_data_analysis.ipynb` - Data analysis fundamentals
- `02_ml_experiment.ipynb` - Complete ML workflow
- `03_visualization_gallery.ipynb` - Visualization showcase

## Using Utility Functions

The `utils/` directory contains reusable functions:

```python
# Data utilities
from utils.data_utils import load_csv, save_csv, get_missing_summary

df = load_csv('data/my_data.csv')
summary = get_missing_summary(df)
save_csv(df, 'data/cleaned_data.csv')

# Model utilities
from utils.model_utils import save_model, evaluate_classifier, ExperimentTracker

save_model(model, 'models/my_model.pkl', metadata={'accuracy': 0.95})
metrics = evaluate_classifier(y_true, y_pred)

tracker = ExperimentTracker()
tracker.log_experiment('exp1', 'RandomForest', params, metrics)

# Visualization utilities
from utils.viz_utils import plot_confusion_matrix, plot_feature_importance

fig = plot_confusion_matrix(cm, labels=['A', 'B'])
fig = plot_feature_importance(feature_names, importances)
```

## Development Workflow

### 1. Explore Data
Start with Jupyter notebooks for exploratory analysis:
```bash
jupyter lab notebooks/examples/01_data_analysis.ipynb
```

### 2. Build Models
Use notebooks or the Streamlit ML Trainer to experiment with models:
```bash
streamlit run streamlit_apps/ml_trainer.py
```

### 3. Create Demos
Build Gradio apps to showcase your models:
```python
import gradio as gr

def predict(input_data):
    # Your model prediction logic
    return result

demo = gr.Interface(fn=predict, inputs="text", outputs="text")
demo.launch()
```

### 4. Visualize Results
Use Streamlit for interactive dashboards:
```bash
streamlit run streamlit_apps/data_dashboard.py
```

## Adding Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install new package
pip install package-name

# Update requirements
pip freeze > requirements.txt
```

## Tips for Small Machines

1. **Use CPU-only versions** of ML frameworks
   - TensorFlow: `pip install tensorflow-cpu`
   - PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

2. **Limit data size** during development
   ```python
   df = df.sample(n=1000)  # Work with samples
   ```

3. **Use lightweight models**
   - Prefer RandomForest over deep learning for small datasets
   - Use `max_depth` and `n_estimators` parameters to control model size

4. **Clear Jupyter outputs** before committing
   ```bash
   jupyter nbconvert --clear-output --inplace notebook.ipynb
   ```

## Common Commands

```bash
# Activate environment
source venv/bin/activate

# Deactivate environment
deactivate

# Run Jupyter Lab
jupyter lab

# Run Gradio app
python gradio_apps/image_classifier.py

# Run Streamlit app
streamlit run streamlit_apps/data_dashboard.py

# Docker commands
docker-compose up              # Start all services
docker-compose up jupyter      # Start only Jupyter
docker-compose down            # Stop all services
docker-compose logs -f         # View logs
```

## Troubleshooting

### Port Already in Use
```bash
# Find process using port
lsof -i :8888

# Kill process
kill -9 <PID>
```

### Module Not Found
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Docker Issues
```bash
# Rebuild containers
docker-compose build --no-cache

# Remove all containers and volumes
docker-compose down -v
```

## Contributing

Feel free to add your own apps, notebooks, and utilities!

1. Create your app/notebook
2. Add documentation
3. Update requirements if needed
4. Test that it works
5. Commit your changes

## Resources

- [Gradio Documentation](https://www.gradio.app/docs)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Jupyter Documentation](https://jupyter.org/documentation)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## License

MIT License - Feel free to use this for your own experiments!

---

**Happy Experimenting!** 🚀
