import logging
from typing import Union, Optional, Literal, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


def normalize_numerical_columns(
    df: pd.DataFrame,
    column: Optional[Union[str, int]] = None,
    method: Literal["standard", "minmax"] = "standard",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Normalize numerical columns in a pandas DataFrame using StandardScaler or MinMaxScaler.

    If `column` is not provided, all numeric columns are normalized.
    If `column` is provided, only that column is normalized (must be numeric).

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str or int, optional): Specific column to normalize.
            If None, all numeric columns are processed.
        method (str): Normalization method. Must be 'standard' (z-score) or 'minmax'. Default is 'standard'.
        inplace (bool): If True, modifies the input DataFrame. Otherwise, returns a copy. Default is False.

    Returns:
        pd.DataFrame: DataFrame with normalized numerical columns.

    Raises:
        TypeError: If input is not a pandas DataFrame.
        ValueError: If specified column does not exist, is not numeric, or if no numeric columns are found.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    df_out = df if inplace else df.copy()

    # Determine columns to normalize
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        raise ValueError("No numeric columns found in the DataFrame. Normalization cannot be applied.")

    if column is not None:
        col_name = df_out.columns[column] if isinstance(column, int) else column
        if col_name not in df_out.columns:
            raise ValueError(f"Column '{col_name}' does not exist in the DataFrame.")
        if col_name not in numeric_cols:
            raise ValueError(f"Column '{col_name}' is not numeric and cannot be normalized.")
        cols_to_normalize = [col_name]
    else:
        cols_to_normalize = numeric_cols

    # Select scaler
    if method == "standard":
        scaler = StandardScaler()
        logger.info("Using StandardScaler for normalization (zero mean, unit variance).")
    elif method == "minmax":
        scaler = MinMaxScaler()
        logger.info("Using MinMaxScaler for normalization (range [0, 1]).")
    else:
        raise ValueError(f"Unsupported normalization method: '{method}'. Use 'standard' or 'minmax'.")

    # Apply scaling
    logger.info(f"Normalizing columns: {cols_to_normalize}")
    df_out[cols_to_normalize] = scaler.fit_transform(df_out[cols_to_normalize])

    logger.info("Normalization completed successfully.")
    return df_out