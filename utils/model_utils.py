"""
Model utility functions
"""

import joblib
import json
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)


def save_model(model: Any, filepath: str, metadata: Optional[Dict] = None) -> None:
    """
    Save model with optional metadata.

    Args:
        model: Trained model
        filepath: Output path
        metadata: Optional metadata dictionary
    """
    try:
        # Save model
        joblib.dump(model, filepath)
        print(f"✅ Model saved to {filepath}")

        # Save metadata if provided
        if metadata:
            meta_path = Path(filepath).with_suffix('.json')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"✅ Metadata saved to {meta_path}")

    except Exception as e:
        print(f"❌ Error saving model: {e}")
        raise


def load_model(filepath: str) -> Any:
    """
    Load saved model.

    Args:
        filepath: Path to model file

    Returns:
        Loaded model
    """
    try:
        model = joblib.load(filepath)
        print(f"✅ Model loaded from {filepath}")

        # Load metadata if exists
        meta_path = Path(filepath).with_suffix('.json')
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            print(f"✅ Metadata loaded from {meta_path}")
            return model, metadata

        return model

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate classification model.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

    return metrics


def evaluate_regressor(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate regression model.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        title: Title to display
    """
    print(f"\n{'=' * 50}")
    print(f"{title}")
    print(f"{'=' * 50}")
    for metric, value in metrics.items():
        print(f"{metric.upper():15s}: {value:.4f}")
    print(f"{'=' * 50}\n")


def compare_models(results: Dict[str, Dict[str, float]], metric: str = 'accuracy') -> str:
    """
    Compare models and return the best one.

    Args:
        results: Dictionary of model names to metrics
        metric: Metric to compare on

    Returns:
        Name of best model
    """
    best_model = max(results.items(), key=lambda x: x[1].get(metric, 0))

    print(f"\nModel Comparison (by {metric}):")
    print(f"{'=' * 50}")
    for model_name, metrics in sorted(results.items(), key=lambda x: x[1].get(metric, 0), reverse=True):
        value = metrics.get(metric, 0)
        indicator = "🏆" if model_name == best_model[0] else "  "
        print(f"{indicator} {model_name:30s}: {value:.4f}")
    print(f"{'=' * 50}")
    print(f"Best model: {best_model[0]}\n")

    return best_model[0]


class ExperimentTracker:
    """Simple experiment tracker for logging experiments."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.experiments = []

    def log_experiment(
        self,
        name: str,
        model_type: str,
        params: Dict,
        metrics: Dict[str, float],
        notes: str = ""
    ) -> None:
        """Log an experiment."""
        import datetime

        experiment = {
            'timestamp': datetime.datetime.now().isoformat(),
            'name': name,
            'model_type': model_type,
            'params': params,
            'metrics': metrics,
            'notes': notes
        }

        self.experiments.append(experiment)

        # Save to file
        log_file = self.log_dir / f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(experiment, f, indent=2, default=str)

        print(f"✅ Experiment logged to {log_file}")

    def get_summary(self) -> None:
        """Print summary of experiments."""
        if not self.experiments:
            print("No experiments logged yet.")
            return

        print(f"\n{'=' * 80}")
        print(f"Experiment Summary ({len(self.experiments)} experiments)")
        print(f"{'=' * 80}")

        for i, exp in enumerate(self.experiments, 1):
            print(f"\n{i}. {exp['name']} ({exp['model_type']})")
            print(f"   Time: {exp['timestamp']}")
            print(f"   Metrics: {exp['metrics']}")
            if exp['notes']:
                print(f"   Notes: {exp['notes']}")

        print(f"\n{'=' * 80}\n")
