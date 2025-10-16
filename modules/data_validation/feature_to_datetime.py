# datetime_conversion.py
import logging
from typing import Union, Optional, Any
import pandas as pd

logger = logging.getLogger(__name__)


def convert_column_to_datetime(
    df: pd.DataFrame,
    column: Union[str, int],
    format: Optional[str] = None,
    errors: str = "raise",
    inplace: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Convert a specified column in a pandas DataFrame to datetime dtype.

    This function uses `pandas.to_datetime` under the hood and supports all its parameters
    via `**kwargs`. Common use cases include parsing strings like '2023-01-01' or timestamps.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str or int): Column name or positional index to convert.
        format (str, optional): strftime format string for parsing (e.g., '%Y-%m-%d').
            If not provided, pandas will attempt to infer the format.
        errors (str): How to handle parsing errors. Options:
            - 'raise': raise an exception (default)
            - 'coerce': set invalid parsing to NaT
            - 'ignore': return the input unchanged for invalid entries
        inplace (bool): If True, modifies the input DataFrame. Otherwise, returns a copy.
        **kwargs: Additional keyword arguments passed to `pandas.to_datetime`.

    Returns:
        pd.DataFrame: DataFrame with the specified column converted to datetime.

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
        ValueError: If `column` does not exist in the DataFrame.
        ValueError: If `errors` is not one of {'raise', 'coerce', 'ignore'}.
        Exception: Any exception raised by `pandas.to_datetime` if `errors='raise'`.

    Examples:
        >>> df = pd.DataFrame({'date_str': ['2023-01-01', '2023-02-01']})
        >>> df_out = convert_column_to_datetime(df, 'date_str')
        >>> df_out['date_str'].dtype
        dtype('<M8[ns]')
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    if errors not in {"raise", "coerce", "ignore"}:
        raise ValueError("Parameter 'errors' must be one of: 'raise', 'coerce', 'ignore'.")

    # Resolve column name
    if isinstance(column, int):
        if column < 0 or column >= df.shape[1]:
            raise ValueError(f"Column index {column} is out of bounds for DataFrame with {df.shape[1]} columns.")
        col_name = df.columns[column]
    else:
        col_name = column
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' does not exist in the DataFrame.")

    logger.info(f"Converting column '{col_name}' to datetime (errors='{errors}')")

    df_out = df if inplace else df.copy()

    try:
        df_out[col_name] = pd.to_datetime(
                                            df_out[col_name],
                                            format=format,
                                            errors=errors, # type: ignore
                                            **kwargs
                                        )# type:ignore
        logger.info(f"Successfully converted column '{col_name}' to datetime dtype: {df_out[col_name].dtype}")
    except Exception as e:
        logger.error(f"Failed to convert column '{col_name}' to datetime: {e}")
        raise

    return df_out