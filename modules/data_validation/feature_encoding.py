# encoding.py
import logging
from typing import Union, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

logger = logging.getLogger(__name__)


def _is_categorical_dtype(series: pd.Series) -> bool:
    """
    Helper to check if a pandas Series has a categorical-like dtype.
    Compatible with pandas >= 2.2.0.
    """
    dtype = series.dtype
    return (
        isinstance(dtype, pd.CategoricalDtype) or
        pd.api.types.is_object_dtype(dtype) or
        (hasattr(pd.StringDtype, '__instancecheck__') and isinstance(dtype, pd.StringDtype))
    )


def apply_one_hot_encoding(
    df: pd.DataFrame,
    column: Union[str, int],
    drop_original: bool = True,
    handle_unknown: str = "ignore",
) -> pd.DataFrame:
    """
    Apply one-hot encoding to a single categorical column in a pandas DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str or int): Column name or positional index.
        drop_original (bool): Whether to drop the original column after encoding. Default is True.
        handle_unknown (str): Strategy for unknown categories during transform. Default is 'ignore'.

    Returns:
        pd.DataFrame: DataFrame with the specified column one-hot encoded.

    Raises:
        ValueError: If the column is not found.
    """
    col_name = df.columns[column] if isinstance(column, int) else column
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in DataFrame.")

    if not _is_categorical_dtype(df[col_name]):
        logger.warning(f"Column '{col_name}' is not categorical (dtype={df[col_name].dtype}). One-hot encoding may not be appropriate.")

    logger.info(f"Applying one-hot encoding to column: '{col_name}'")

    # Handle NaN by converting to string "NaN" to avoid encoder errors
    data = df[[col_name]].astype(str).fillna("NaN")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown=handle_unknown)
    encoded_array = encoder.fit_transform(data)

    categories = encoder.categories_[0]
    new_col_names = [f"{col_name}_{cat}" for cat in categories]
    encoded_df = pd.DataFrame(encoded_array, columns=new_col_names, index=df.index)

    if drop_original:
        result = pd.concat([df.drop(columns=[col_name]), encoded_df], axis=1)
    else:
        result = pd.concat([df, encoded_df], axis=1)

    logger.info(f"One-hot encoding completed. Added {len(new_col_names)} new columns.")
    return result


def apply_label_encoding(
    df: pd.DataFrame,
    column: Union[str, int],
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    Apply label (ordinal) encoding to a single categorical column in a pandas DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str or int): Column name or positional index.
        drop_original (bool): Whether to drop the original column. Default is True.

    Returns:
        pd.DataFrame: DataFrame with the specified column label-encoded as integers.

    Raises:
        ValueError: If the column is not found.
    """
    col_name = df.columns[column] if isinstance(column, int) else column
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in DataFrame.")

    if not _is_categorical_dtype(df[col_name]):
        logger.warning(f"Column '{col_name}' is not categorical (dtype={df[col_name].dtype}). Label encoding may not be appropriate.")

    logger.info(f"Applying label encoding to column: '{col_name}'")

    series = df[col_name].astype(str).fillna("NaN")
    encoder = LabelEncoder()
    encoded_values = encoder.fit_transform(series)

    result = df.copy()
    new_col_name = f"{col_name}_encoded"
    result[new_col_name] = encoded_values

    if drop_original:
        result = result.drop(columns=[col_name])

    logger.info(f"Label encoding completed. Mapped {len(encoder.classes_)} unique values.")
    return result


def encode_categorical_column(
    df: pd.DataFrame,
    column: Union[str, int],
    method: str = "one_hot",
    **kwargs,
) -> pd.DataFrame:
    """
    High-level function to encode a categorical column using the specified method.

    Validates that:
        - Input is a pandas DataFrame
        - Column exists
        - Column is of a categorical-like type

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str or int): Column name or index.
        method (str): Encoding method. Must be 'one_hot' or 'label'. Default is 'one_hot'.
        **kwargs: Additional arguments passed to the underlying encoder function.

    Returns:
        pd.DataFrame: Encoded DataFrame.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If column is invalid or method is unsupported.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    col_name = df.columns[column] if isinstance(column, int) else column
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' does not exist in the DataFrame.")

    if not _is_categorical_dtype(df[col_name]):
        logger.warning(f"Column '{col_name}' is not of categorical type (dtype={df[col_name].dtype}). Proceeding anyway.")

    logger.info(f"Encoding column '{col_name}' using method: '{method}'")

    if method == "one_hot":
        return apply_one_hot_encoding(df, column, **kwargs)
    elif method == "label":
        return apply_label_encoding(df, column, **kwargs)
    else:
        raise ValueError(f"Unsupported encoding method: '{method}'. Use 'one_hot' or 'label'.")