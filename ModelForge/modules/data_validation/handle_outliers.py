import logging
from typing import Union, Literal
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def handle_outliers(
    df: pd.DataFrame,
    column: Union[str, int],
    method: Literal["iqr", "zscore"] = "iqr",
    handling: Literal["remove", "clip", "impute"] = "remove",
    impute_strategy: Literal["mean", "median"] = "median",
    threshold: float = 1.5,
    z_threshold: float = 3.0,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Detect and handle outliers in a numeric column of a pandas DataFrame.

    Supported detection methods:
        - 'iqr': Uses Interquartile Range (Q1 - threshold*IQR, Q3 + threshold*IQR)
        - 'zscore': Uses Z-score (|z| > z_threshold)

    Supported handling strategies:
        - 'remove': Drop rows with outliers
        - 'clip': Cap outliers to the nearest non-outlier boundary
        - 'impute': Replace outliers with mean or median of non-outlier values

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str or int): Column name or index to process.
        method (str): Outlier detection method. Default is 'iqr'.
        handling (str): How to handle detected outliers. Default is 'clip'.
        impute_strategy (str): Strategy for imputation ('mean' or 'median'). Used only if handling='impute'.
        threshold (float): IQR multiplier (e.g., 1.5). Used only if method='iqr'.
        z_threshold (float): Z-score cutoff (e.g., 3.0). Used only if method='zscore'.
        inplace (bool): If True, modifies the input DataFrame. Otherwise, returns a copy.

    Returns:
        pd.DataFrame: DataFrame with outliers handled as specified.

    Raises:
        TypeError: If input is not a pandas DataFrame.
        ValueError: If column is not numeric or does not exist.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    col_name = df.columns[column] if isinstance(column, int) else column
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' does not exist in the DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[col_name]):
        raise ValueError(f"Column '{col_name}' is not numeric and cannot be processed for outliers.")

    df_out = df if inplace else df.copy()
    series = df_out[col_name].copy()

    # Step 1: Detect outliers
    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        logger.info(f"IQR method: bounds = [{lower_bound:.4f}, {upper_bound:.4f}]")

    elif method == "zscore":
        z_scores = np.abs((series - series.mean()) / series.std(ddof=0))
        outlier_mask = z_scores > z_threshold
        logger.info(f"Z-score method: threshold = {z_threshold}")

    else:
        raise ValueError("Method must be 'iqr' or 'zscore'.")

    n_outliers = outlier_mask.sum()
    logger.info(f"Detected {n_outliers} outliers in column '{col_name}' using method '{method}'.")

    if n_outliers == 0:
        logger.info("No outliers to handle.")
        return df_out

    # Step 2: Handle outliers
    if handling == "remove":
        df_out = df_out[~outlier_mask].copy()
        logger.info(f"Removed {n_outliers} rows containing outliers.")

    elif handling == "clip":
        if method == "iqr":
            df_out[col_name] = series.clip(lower=lower_bound, upper=upper_bound)
            print('we are here')
        else:  # zscore → use percentiles matching z=±3 (~99.7%)
            lower_clip = series.quantile(0.0015)
            upper_clip = series.quantile(0.9985)
            df_out[col_name] = series.clip(lower=lower_clip, upper=upper_clip)
            print('Ormwe are here')
        logger.info(f"Clipped {n_outliers} outliers to boundary values.")

    elif handling == "impute":
        non_outlier_values = series[~outlier_mask]
        if impute_strategy == "mean":
            fill_val = non_outlier_values.mean()
        else:
            fill_val = non_outlier_values.median()
        df_out.loc[outlier_mask, col_name] = fill_val
        logger.info(f"Imputed {n_outliers} outliers with {impute_strategy}: {fill_val:.4f}")

    else:
        raise ValueError("Handling must be 'remove', 'clip', or 'impute'.")

    return df_out