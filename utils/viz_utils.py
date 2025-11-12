"""
Visualization utility functions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple


def set_plot_style(style: str = 'whitegrid', context: str = 'notebook') -> None:
    """
    Set plotting style.

    Args:
        style: Seaborn style
        context: Seaborn context
    """
    sns.set_style(style)
    sns.set_context(context)
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def plot_distribution(
    data: pd.Series,
    title: str = "Distribution",
    bins: int = 30,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot distribution with histogram and KDE.

    Args:
        data: Data to plot
        title: Plot title
        bins: Number of bins
        figsize: Figure size

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Histogram
    ax.hist(data, bins=bins, alpha=0.6, edgecolor='black', density=True, label='Histogram')

    # KDE
    data.plot.kde(ax=ax, linewidth=2, label='KDE')

    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    title: str = "Correlation Matrix",
    figsize: Tuple[int, int] = (10, 8),
    annot: bool = True
) -> plt.Figure:
    """
    Plot correlation matrix heatmap.

    Args:
        df: DataFrame with numeric columns
        title: Plot title
        figsize: Figure size
        annot: Whether to annotate cells

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate correlation
    corr = df.select_dtypes(include=[np.number]).corr()

    # Plot heatmap
    sns.heatmap(
        corr,
        annot=annot,
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )

    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 15,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot feature importance.

    Args:
        feature_names: List of feature names
        importances: Array of importance values
        top_n: Number of top features to show
        title: Plot title
        figsize: Figure size

    Returns:
        Figure object
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(range(len(indices)), importances[indices])
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_title(title)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Importance')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    normalize: bool = False
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        labels: Class labels
        title: Plot title
        figsize: Figure size
        normalize: Whether to normalize

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        xticklabels=labels,
        yticklabels=labels
    )

    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    return fig


def plot_learning_curves(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    title: str = "Learning Curves",
    ylabel: str = "Score",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot learning curves.

    Args:
        train_scores: Training scores
        val_scores: Validation scores
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_scores) + 1)

    ax.plot(epochs, train_scores, 'o-', label='Training', linewidth=2)
    ax.plot(epochs, val_scores, 's-', label='Validation', linewidth=2)

    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot predictions vs actual values.

    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        figsize: Figure size

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_title(title)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_time_series(
    df: pd.DataFrame,
    date_column: str,
    value_columns: List[str],
    title: str = "Time Series",
    figsize: Tuple[int, int] = (14, 6),
    moving_average: Optional[int] = None
) -> plt.Figure:
    """
    Plot time series data.

    Args:
        df: DataFrame with time series data
        date_column: Name of date column
        value_columns: List of value columns to plot
        title: Plot title
        figsize: Figure size
        moving_average: Window size for moving average

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    for col in value_columns:
        ax.plot(df[date_column], df[col], label=col, alpha=0.7, linewidth=2)

        if moving_average:
            ma = df[col].rolling(window=moving_average).mean()
            ax.plot(df[date_column], ma, label=f'{col} (MA{moving_average})',
                   linestyle='--', linewidth=2)

    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def save_figure(fig: plt.Figure, filepath: str, dpi: int = 300) -> None:
    """
    Save figure to file.

    Args:
        fig: Figure to save
        filepath: Output path
        dpi: Resolution
    """
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"✅ Figure saved to {filepath}")
    except Exception as e:
        print(f"❌ Error saving figure: {e}")
        raise
