# Gradio Apps

This directory contains Gradio demo applications for quick ML model interfaces.

## Available Apps

1. **image_classifier.py** - Image classification demo
2. **text_analyzer.py** - Text analysis and statistics
3. **simple_ml_demo.py** - Simple ML model with feature importance

## Running an App

```bash
# Run a specific app
python gradio_apps/image_classifier.py

# Or
python gradio_apps/text_analyzer.py

# Or
python gradio_apps/simple_ml_demo.py
```

The app will start a web server (usually on port 7860-7862) and provide a URL to access the interface.

## Creating Your Own App

```python
import gradio as gr

def my_function(input_text):
    return f"You said: {input_text}"

demo = gr.Interface(
    fn=my_function,
    inputs=gr.Textbox(label="Input"),
    outputs=gr.Textbox(label="Output"),
    title="My Demo"
)

if __name__ == "__main__":
    demo.launch()
```

Learn more at: https://www.gradio.app/docs
