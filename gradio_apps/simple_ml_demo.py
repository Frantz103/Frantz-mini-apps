"""
Simple ML Model Demo
A lightweight demo showing how to train and use a scikit-learn model with Gradio.
"""

import gradio as gr
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import io
from PIL import Image


# Train a simple model (you can replace this with your own model)
def train_model():
    """Train a simple classifier on synthetic data."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, accuracy


# Global model
MODEL, ACCURACY = train_model()


def predict_with_features(*features):
    """Make a prediction given input features."""
    try:
        # Convert inputs to float array
        feature_array = np.array([float(f) for f in features]).reshape(1, -1)

        # Make prediction
        prediction = MODEL.predict(feature_array)[0]
        probability = MODEL.predict_proba(feature_array)[0]

        result = f"""
## Prediction Result

**Predicted Class:** {prediction}
**Class 0 Probability:** {probability[0]:.2%}
**Class 1 Probability:** {probability[1]:.2%}

**Model Accuracy:** {ACCURACY:.2%}
        """

        return result

    except Exception as e:
        return f"Error: {str(e)}"


def visualize_feature_importance():
    """Visualize feature importances."""
    importances = MODEL.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.tight_layout()

    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()

    return img


# Create Gradio interface
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 🤖 Simple ML Model Demo")
    gr.Markdown(f"This demo uses a Random Forest classifier trained on synthetic data (Accuracy: {ACCURACY:.2%})")

    with gr.Tab("Make Predictions"):
        gr.Markdown("### Enter 20 feature values to make a prediction")

        with gr.Row():
            with gr.Column():
                # Create 20 input fields
                feature_inputs = [
                    gr.Number(label=f"Feature {i+1}", value=0.0)
                    for i in range(20)
                ]

                predict_btn = gr.Button("Predict", variant="primary")

            with gr.Column():
                prediction_output = gr.Markdown()

        predict_btn.click(
            fn=predict_with_features,
            inputs=feature_inputs,
            outputs=prediction_output,
        )

    with gr.Tab("Feature Importance"):
        gr.Markdown("### Feature Importance Visualization")
        importance_btn = gr.Button("Show Feature Importance")
        importance_plot = gr.Image(label="Feature Importance")

        importance_btn.click(
            fn=visualize_feature_importance,
            inputs=None,
            outputs=importance_plot,
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
    )
