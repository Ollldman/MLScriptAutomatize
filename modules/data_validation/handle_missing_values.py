# data_cleaning.py
import logging
from typing import Union, Optional, Literal, Any, List
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

logger = logging.getLogger(__name__)


def handle_missing_numeric_values(
    df: pd.DataFrame,
    strategy: Literal["mean", "median", "constant"] = "mean",
    fill_value: Optional[Union[float, int]] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Impute missing values in numeric columns of a pandas DataFrame.

    Supported strategies:
        - "mean": replace with column mean
        - "median": replace with column median
        - "constant": replace with a provided scalar value

    Args:
        df (pd.DataFrame): Input DataFrame.
        strategy (str): Imputation strategy. Must be one of {"mean", "median", "constant"}.
        fill_value (float | int, optional): Value to use when strategy="constant".
        columns (List[str], optional): Subset of numeric columns to process.
            If None, all numeric columns are used.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with missing values imputed in specified numeric columns.

    Raises:
        ValueError: If strategy is "constant" but fill_value is not provided.
        TypeError: If non-numeric columns are passed explicitly.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty.")
        return df.copy()

    df_out = df.copy()

    # Determine numeric columns to process
    all_numeric_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
    if columns is None:
        cols_to_process = all_numeric_cols
    else:
        # Validate that all specified columns are numeric
        invalid_cols = [col for col in columns if col not in all_numeric_cols]
        if invalid_cols:
            raise TypeError(f"Columns {invalid_cols} are not numeric and cannot be processed by this function.")
        cols_to_process = columns

    if not cols_to_process:
        logger.info("No numeric columns to process.")
        return df_out

    missing_before = df_out[cols_to_process].isnull().sum().sum()
    if missing_before == 0:
        logger.info("No missing values found in numeric columns.")
        return df_out

    logger.info(f"Imputing {missing_before} missing values in numeric columns: {cols_to_process}")

    if strategy == "constant":
        if fill_value is None:
            raise ValueError("Parameter 'fill_value' must be provided when strategy='constant'.")
        imputer = SimpleImputer(strategy="constant", fill_value=fill_value)
    else:
        imputer = SimpleImputer(strategy=strategy)

    df_out[cols_to_process] = imputer.fit_transform(df_out[cols_to_process])

    missing_after = df_out[cols_to_process].isnull().sum().sum()
    logger.info(f"Missing values after imputation in numeric columns: {missing_after}")
    return df_out


def handle_missing_categorical_values(
    df: pd.DataFrame,
    strategy: Literal["most_frequent", "constant"] = "most_frequent",
    fill_value: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Impute missing values in categorical (non-numeric) columns of a pandas DataFrame.

    Supported strategies:
        - "most_frequent": replace with the most common value in the column
        - "constant": replace with a provided string value

    Args:
        df (pd.DataFrame): Input DataFrame.
        strategy (str): Imputation strategy. Must be one of {"most_frequent", "constant"}.
        fill_value (str, optional): Value to use when strategy="constant".
        columns (List[str], optional): Subset of categorical columns to process.
            If None, all non-numeric columns are used.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with missing values imputed in specified categorical columns.

    Raises:
        ValueError: If strategy is "constant" but fill_value is not provided.
        TypeError: If numeric columns are passed explicitly.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty.")
        return df.copy()

    df_out = df.copy()

    # Identify non-numeric (categorical-like) columns
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    all_categorical_cols = [col for col in df_out.columns if col not in numeric_cols]

    if columns is None:
        cols_to_process = all_categorical_cols
    else:
        # Validate that specified columns are not numeric
        invalid_cols = [col for col in columns if col in numeric_cols]
        if invalid_cols:
            raise TypeError(f"Columns {invalid_cols} are numeric and should be handled by handle_missing_numeric_values().")
        cols_to_process = columns

    if not cols_to_process:
        logger.info("No categorical columns to process.")
        return df_out

    missing_before = df_out[cols_to_process].isnull().sum().sum()
    if missing_before == 0:
        logger.info("No missing values found in categorical columns.")
        return df_out

    logger.info(f"Imputing {missing_before} missing values in categorical columns: {cols_to_process}")

    if strategy == "constant":
        if fill_value is None:
            raise ValueError("Parameter 'fill_value' must be provided when strategy='constant'.")
        imputer = SimpleImputer(strategy="constant", fill_value=fill_value)
    else:  # most_frequent
        imputer = SimpleImputer(strategy="most_frequent")

    # sklearn returns object dtype; preserve as string if possible
    print(f'\n \n {cols_to_process} \n\n')
    df_out[cols_to_process] = imputer.fit_transform(df_out[cols_to_process])
    print(f'\n \n{df_out}\n\n')
    missing_after = df_out[cols_to_process].isnull().sum().sum()
    logger.info(f"Missing values after imputation in categorical columns: {missing_after}")
    return df_out


def check_dataset(df: pd.DataFrame) -> str:
    """
    Analyze a pandas DataFrame and generate a diagnostic report about missing values,
    distinguishing between numeric and categorical columns.

    Returns a human-readable string with counts and column names containing missing values.

    Args:
        df (pd.DataFrame): Input DataFrame to analyze.

    Returns:
        str: Diagnostic message describing missing data distribution.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty.")
        return "Dataset is empty."

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    missing_numeric = {}
    missing_categorical = {}

    if numeric_cols:
        missing_counts_num = df[numeric_cols].isnull().sum()
        missing_numeric = missing_counts_num[missing_counts_num > 0].to_dict()

    if categorical_cols:
        missing_counts_cat = df[categorical_cols].isnull().sum()
        missing_categorical = missing_counts_cat[missing_counts_cat > 0].to_dict()

    total_missing_num = sum(missing_numeric.values())
    total_missing_cat = sum(missing_categorical.values())

    parts = []
    if total_missing_num > 0:
        cols_str = ", ".join(missing_numeric.keys())
        parts.append(f"{total_missing_num} missing values in numeric columns: [{cols_str}]")
    if total_missing_cat > 0:
        cols_str = ", ".join(missing_categorical.keys())
        parts.append(f"{total_missing_cat} missing values in categorical columns: [{cols_str}]")

    if not parts:
        report = "No missing values detected."
    else:
        report = "Detected: " + "; ".join(parts)

    logger.info(f"Dataset check result: {report}")
    return report


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: Optional[Literal["mean", "median", "constant"]] = None,
    numeric_fill_value: Optional[Union[float, int]] = None,
    numeric_columns: Optional[List[str]] = None,
    categorical_strategy: Optional[Literal["most_frequent", "constant"]] = None,
    categorical_fill_value: Optional[str] = None,
    categorical_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    High-level function to handle missing values in a pandas DataFrame by delegating
    to specialized numeric and categorical imputation functions.

    If a strategy is not provided for a data type (numeric/categorical), that part is skipped.

    Args:
        df (pd.DataFrame): Input DataFrame.
        numeric_strategy (str, optional): Strategy for numeric columns.
        numeric_fill_value (float|int, optional): Fill value if numeric_strategy='constant'.
        numeric_columns (List[str], optional): Subset of numeric columns to process.
        categorical_strategy (str, optional): Strategy for categorical columns.
        categorical_fill_value (str, optional): Fill value if categorical_strategy='constant'.
        categorical_columns (List[str], optional): Subset of categorical columns to process.

    Returns:
        pd.DataFrame: Cleaned DataFrame with missing values handled as specified.

    Raises:
        ValueError: If strategy='constant' but fill_value is not provided.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty. Returning as-is.")
        return df.copy()

    df_out = df.copy()

    # Handle numeric columns if strategy is specified
    if numeric_strategy is not None:
        logger.info(f"Applying numeric imputation strategy: {numeric_strategy}")
        df_out = handle_missing_numeric_values(
            df_out,
            strategy=numeric_strategy,
            fill_value=numeric_fill_value,
            columns=numeric_columns,
        )

    # Handle categorical columns if strategy is specified
    if categorical_strategy is not None:
        logger.info(f"Applying categorical imputation strategy: {categorical_strategy}")
        df_out = handle_missing_categorical_values(
            df_out,
            strategy=categorical_strategy,
            fill_value=categorical_fill_value,
            columns=categorical_columns,
        )

    # Final report
    final_info = df_out.info(buf=None, verbose=True, show_counts=True)
    logger.info("Missing value handling completed. Final DataFrame info logged.")

    # Note: df.info() prints to stdout; we don't capture it as string here for simplicity.
    # If needed, use StringIO to capture, but typically logging shape/stats is enough:
    logger.info(f"Result shape: {df_out.shape}, nulls total: {df_out.isnull().sum().sum()}")

    return df_out