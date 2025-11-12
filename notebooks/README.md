# Jupyter Notebooks

This directory contains Jupyter notebooks for exploratory analysis and ML experiments.

## Directory Structure

```
notebooks/
├── examples/           # Example notebooks to get started
│   ├── 01_data_analysis.ipynb
│   ├── 02_ml_experiment.ipynb
│   └── 03_visualization_gallery.ipynb
└── README.md
```

## Example Notebooks

1. **01_data_analysis.ipynb** - Data analysis basics with pandas and visualization
2. **02_ml_experiment.ipynb** - Complete ML workflow from data prep to evaluation
3. **03_visualization_gallery.ipynb** - Showcase of matplotlib, seaborn, and plotly

## Running Jupyter

### Start Jupyter Lab (recommended)

```bash
jupyter lab
```

### Start Jupyter Notebook

```bash
jupyter notebook
```

### Run from specific directory

```bash
jupyter lab notebooks/
```

## Tips

- **Jupyter Lab** is the modern interface with more features
- **Jupyter Notebook** is the classic interface, lighter weight
- Press `Shift+Enter` to run a cell
- Use `%%time` at the start of a cell to measure execution time
- Use `?` after a function to see its documentation

## Best Practices

1. **Organize your work**: Create separate notebooks for different experiments
2. **Clear outputs**: Clear output before committing to git
3. **Use markdown**: Document your analysis with markdown cells
4. **Save frequently**: Jupyter autosaves, but manual saves are safer
5. **Restart kernel**: Restart kernel and run all cells to verify reproducibility

## Converting Notebooks

```bash
# Convert to Python script
jupyter nbconvert --to script notebook.ipynb

# Convert to HTML
jupyter nbconvert --to html notebook.ipynb

# Convert to PDF (requires LaTeX)
jupyter nbconvert --to pdf notebook.ipynb
```

## Keyboard Shortcuts

- `Shift+Enter`: Run cell and move to next
- `Ctrl+Enter`: Run cell in place
- `a`: Insert cell above
- `b`: Insert cell below
- `dd`: Delete cell
- `m`: Change cell to markdown
- `y`: Change cell to code

Learn more at: https://jupyter.org/
