import logging
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def compute_descriptive_statistics(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    percentiles: Optional[List[float]] = None,
    include_all: bool = True,
) -> Dict[str, Dict[str, Union[float, List[float], str, None]]]:
    """
    Compute descriptive statistics for numeric columns in a pandas DataFrame
    using pandas' built-in `.describe()` method for robustness and performance.

    For each selected numeric column, the function returns:
        - count
        - mean
        - std
        - min
        - max
        - percentiles (e.g., 25%, 50%, 75% by default)
        - mode (optional, computed separately if `include_all=True`)

    Non-numeric columns are automatically excluded.

    Args:
        df (pd.DataFrame): Input DataFrame after preprocessing.
        columns (List[str], optional): List of column names to analyze.
            If None, all numeric columns are used.
        percentiles (List[float], optional): List of percentiles to include (values in [0, 1]).
            Default: [0.25, 0.5, 0.75] → 25th, 50th (median), 75th percentiles.
        include_all (bool): If True, also computes mode and includes it in output.
            Note: mode may be slow for high-cardinality data.

    Returns:
        Dict[str, Dict[str, Union[float, List[float], str, None]]]: A nested dictionary where:
            - Top-level keys: column names
            - Inner keys: 'count', 'mean', 'std', 'min', 'max', 'percentiles', 'mode' (if enabled)
            - 'percentiles' is a dict like {'25%': 10.0, '50%': 20.0, '75%': 30.0}
            - 'mode' is a list (may be empty)

    Raises:
        TypeError: If input is not a pandas DataFrame.
        ValueError: If no numeric columns are available.

    Example:
        >>> df = pd.DataFrame({'A': [1, 2, 2, 4]})
        >>> stats = compute_descriptive_statistics(df)
        >>> print(stats['A']['mean'])
        2.25
        >>> print(stats['A']['percentiles']['50%'])
        2.0
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    if df.empty:
        logger.warning("Input DataFrame is empty.")
        raise ValueError("No columns found in the DataFrame.")

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found in the DataFrame.")

    if columns is not None:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        non_numeric = [col for col in columns if col not in numeric_cols]
        if non_numeric:
            logger.warning(f"Non-numeric columns skipped: {non_numeric}")
        cols_to_analyze = [col for col in columns if col in numeric_cols]
    else:
        cols_to_analyze = numeric_cols

    if not cols_to_analyze:
        raise ValueError("No valid numeric columns to analyze.")

    # Set default percentiles
    if percentiles is None:
        percentiles = [0.25, 0.5, 0.75]
    else:
        # Validate percentiles
        if not all(0 <= p <= 1 for p in percentiles):
            raise ValueError("All percentiles must be in the range [0, 1].")

    logger.info(f"Computing statistics for columns: {cols_to_analyze}")
    logger.info(f"Percentiles: {[f'{p:.0%}' for p in percentiles]}")

    statistics: Dict[str, Dict[str, Any]] = {}

    # Use describe for core stats
    desc = df[cols_to_analyze].describe(percentiles=percentiles)

    for col in cols_to_analyze:
        col_stats = {
            "count": int(desc[col]["count"]),
            "mean": float(desc[col]["mean"]),
            "std": float(desc[col]["std"]) if desc[col]["count"] > 1 else 0.0,
            "min": float(desc[col]["min"]),
            "max": float(desc[col]["max"]),
        }

        # Extract percentiles as a sub-dict
        percentile_labels = [f"{p:.0%}" for p in percentiles]
        col_stats["percentiles"] = {
            label: float(desc[col][label]) for label in percentile_labels
        }

        # Optional: mode
        if include_all:
            mode_series = df[col].dropna().mode()
            col_stats["mode"] = mode_series.tolist() if not mode_series.empty else []

        statistics[col] = col_stats

    logger.info("Descriptive statistics computation completed using pandas.describe().")
    return statistics