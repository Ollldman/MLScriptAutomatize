import logging
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

logger = logging.getLogger(__name__)


def analyze_time_series(
    df: pd.DataFrame,
    value_column: str,
    datetime_column: Optional[str] = None,
    freq: Optional[str] = None,
    model: str = "additive",
) -> Dict[str, Any]:
    """
    Perform comprehensive trend and seasonality analysis on a time series.

    This function:
        - Validates and prepares the time series
        - Computes basic statistics (mean, std, min, max, etc.)
        - Tests for stationarity (Augmented Dickey-Fuller test)
        - Decomposes the series into trend, seasonal, and residual components
        - Returns a structured report with key metrics

    Args:
        df (pd.DataFrame): Input DataFrame containing time series data.
        value_column (str): Name of the numeric column to analyze.
        datetime_column (str, optional): Name of the datetime column.
            If None, the DataFrame index must be a DatetimeIndex.
        freq (str, optional): Frequency string (e.g., 'D', 'H', 'M').
            Required if the index/column is not already periodic.
        model (str): Decomposition model. Must be 'additive' or 'multiplicative'.
            Default is 'additive'.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'basic_stats': mean, std, min, max, count
            - 'stationarity': ADF statistic, p-value, is_stationary (bool)
            - 'trend_strength': R²-like measure of trend presence
            - 'seasonality_strength': R²-like measure of seasonality presence
            - 'decomposition': dict with 'trend', 'seasonal', 'resid' (as lists)
            - 'missing_values': count of NaN in value_column

    Raises:
        ValueError: If inputs are invalid or time series cannot be prepared.
        TypeError: If df is not a pandas DataFrame.

    Example:
        >>> df = pd.read_csv("sales.csv", parse_dates=["date"], index_col="date")
        >>> report = analyze_time_series(df, value_column="sales")
        >>> print(report["stationarity"]["is_stationary"])
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    if value_column not in df.columns:
        raise ValueError(f"Value column '{value_column}' not found in DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[value_column]):
        raise ValueError(f"Value column '{value_column}' must be numeric.")

    # Prepare datetime index
    if datetime_column is not None:
        if datetime_column not in df.columns:
            raise ValueError(f"Datetime column '{datetime_column}' not found.")
        df_ts = df[[datetime_column, value_column]].copy()
        df_ts = df_ts.set_index(pd.to_datetime(df_ts[datetime_column]))
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex if datetime_column is not provided.")
        df_ts = df[[value_column]].copy()

    df_ts = df_ts.asfreq(freq) if freq else df_ts
    series = df_ts[value_column].dropna()

    if series.empty:
        raise ValueError("No valid data after dropping NaN values.")

    logger.info(f"Analyzing time series with {len(series)} observations.")

    # 1. Basic statistics
    basic_stats = {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
        "min": float(series.min()),
        "max": float(series.max()),
        "missing_values": int(df[value_column].isna().sum()),
    }

    # 2. Stationarity test (ADF)
    try:
        adf_result = adfuller(series, autolag="AIC")
        adf_stat, p_value = adf_result[0], adf_result[1]
        is_stationary = p_value < 0.05
        stationarity = {
            "adf_statistic": float(adf_stat),
            "p_value": float(p_value),
            "is_stationary": is_stationary,
            "used_lags": adf_result[2],
            "n_observations": adf_result[3],
        }
    except (ValueError, np.linalg.LinAlgError) as e:
        logger.warning(f"ADF test failed: {e}")
        stationarity = {
            "adf_statistic": None,
            "p_value": None,
            "is_stationary": False,
            "used_lags": None,
            "n_observations": None,
        }

    # 3. Seasonal decomposition
    decomposition = {"trend": [], "seasonal": [], "resid": []}
    trend_strength = 0.0
    seasonality_strength = 0.0

    try:
        decomp = seasonal_decompose(series, model=model, extrapolate_trend='freq') # type:ignore
        decomposition = {
            "trend": decomp.trend.dropna().tolist(),
            "seasonal": decomp.seasonal.dropna().tolist(),
            "resid": decomp.resid.dropna().tolist(),
        }

        # Strength measures (similar to Hyndman's approach)
        resid_var = np.var(decomp.resid.dropna())
        if model == "additive":
            total_var = np.var(series)
            trend_var = np.var(decomp.trend.dropna())
            seasonal_var = np.var(decomp.seasonal.dropna())
        else:  # multiplicative
            total_var = np.var(np.log(series))
            log_trend = np.log(decomp.trend.dropna().replace(0, np.nan)).dropna()
            log_seasonal = np.log(decomp.seasonal.dropna().replace(0, np.nan)).dropna()
            trend_var = np.var(log_trend)
            seasonal_var = np.var(log_seasonal)

        if total_var > 0:
            trend_strength = max(0.0, 1 - resid_var / (resid_var + trend_var)) if trend_var > 0 else 0.0
            seasonality_strength = max(0.0, 1 - resid_var / (resid_var + seasonal_var)) if seasonal_var > 0 else 0.0

    except (ValueError, np.linalg.LinAlgError) as e:
        logger.warning(f"Seasonal decomposition failed: {e}")

    result = {
        "basic_stats": basic_stats,
        "stationarity": stationarity,
        "trend_strength": float(trend_strength),
        "seasonality_strength": float(seasonality_strength),
        "decomposition": decomposition,
    }

    logger.info("Time series analysis completed successfully.")
    return result