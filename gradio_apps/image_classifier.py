"""
Simple Image Classifier Demo
A lightweight example showing how to build an image classification interface with Gradio.
"""

import gradio as gr
import numpy as np
from PIL import Image
import io


def classify_image(image):
    """
    Placeholder image classifier.
    Replace this with your actual ML model for real classification.
    """
    if image is None:
        return {"Error": 1.0}

    # Convert to RGB if needed
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype('uint8'), 'RGB')
    else:
        img = image

    # Get image properties
    width, height = img.size

    # Dummy classification - replace with actual model
    # Example: model.predict(preprocess(image))
    predictions = {
        "Cat": np.random.random(),
        "Dog": np.random.random(),
        "Bird": np.random.random(),
        "Fish": np.random.random(),
        "Other": np.random.random(),
    }

    # Normalize to sum to 1
    total = sum(predictions.values())
    predictions = {k: v/total for k, v in predictions.items()}

    return predictions


# Create Gradio interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="🖼️ Image Classifier Demo",
    description="Upload an image to classify it. (This is a demo with random predictions - replace with your actual model!)",
    examples=[
        # Add paths to example images if you have any
    ],
    theme="soft",
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True to create a public link
    )
