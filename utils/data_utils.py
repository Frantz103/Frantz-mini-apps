"""
Data utility functions
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load CSV file with error handling.

    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        DataFrame
    """
    try:
        df = pd.read_csv(filepath, **kwargs)
        print(f"✅ Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        raise


def save_csv(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    """
    Save DataFrame to CSV with confirmation.

    Args:
        df: DataFrame to save
        filepath: Output path
        **kwargs: Additional arguments for df.to_csv
    """
    try:
        df.to_csv(filepath, index=False, **kwargs)
        print(f"✅ Saved {len(df)} rows to {filepath}")
    except Exception as e:
        print(f"❌ Error saving to {filepath}: {e}")
        raise


def train_test_split_by_time(
    df: pd.DataFrame,
    date_column: str,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data chronologically.

    Args:
        df: DataFrame with time series data
        date_column: Name of date column
        test_size: Fraction for test set

    Returns:
        Tuple of (train_df, test_df)
    """
    df = df.sort_values(date_column)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"Train: {len(train_df)} rows ({train_df[date_column].min()} to {train_df[date_column].max()})")
    print(f"Test: {len(test_df)} rows ({test_df[date_column].min()} to {test_df[date_column].max()})")

    return train_df, test_df


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary of missing values.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with missing value statistics
    """
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)

    summary = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })

    summary = summary[summary['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

    return summary


def remove_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers using IQR method.

    Args:
        df: Input DataFrame
        column: Column to check for outliers
        multiplier: IQR multiplier (default 1.5)

    Returns:
        DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()

    outliers_removed = len(df) - len(df_clean)
    print(f"Removed {outliers_removed} outliers from {column} ({outliers_removed/len(df)*100:.2f}%)")

    return df_clean


def generate_sample_timeseries(
    n_samples: int = 1000,
    start_date: str = '2023-01-01',
    freq: str = 'D',
    trend: bool = True,
    seasonality: bool = True,
    noise_level: float = 0.1
) -> pd.DataFrame:
    """
    Generate sample time series data.

    Args:
        n_samples: Number of samples
        start_date: Start date
        freq: Frequency ('D', 'H', etc.)
        trend: Add trend component
        seasonality: Add seasonal component
        noise_level: Noise level

    Returns:
        DataFrame with time series data
    """
    dates = pd.date_range(start=start_date, periods=n_samples, freq=freq)

    # Base value
    value = np.zeros(n_samples)

    # Add trend
    if trend:
        value += np.linspace(0, 50, n_samples)

    # Add seasonality
    if seasonality:
        seasonal_period = min(365, n_samples // 3) if freq == 'D' else 24
        value += 10 * np.sin(2 * np.pi * np.arange(n_samples) / seasonal_period)

    # Add noise
    value += np.random.normal(0, noise_level * value.std(), n_samples)

    df = pd.DataFrame({
        'date': dates,
        'value': value + 100  # Offset to make positive
    })

    return df
