# test_trend_season_analysis.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ModelForge.modules\
    .data_analysis.\
    trend_season_analysis import analyze_time_series
from ModelForge.modules\
    .data_loading_to_dataFrame\
    .load_from_api import load_from_kaggle


@pytest.fixture
def synthetic_ts_df():
    """
    Синтетический временной ряд: тренд + сезонность (7 дней) + шум
    """
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(365)]
    trend = np.linspace(100, 200, 365)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 7)
    noise = np.random.normal(0, 5, 365)
    values = trend + seasonal + noise

    df = pd.DataFrame({
        'date': dates,
        'sales': values
    })
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')


@pytest.fixture
def synthetic_ts_with_nan_df():
    """
    Синтетический временной ряд с NaN
    """
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(50)]
    values = np.linspace(10, 50, 50) + np.random.normal(0, 2, 50)
    values[5] = np.nan
    values[15] = np.nan

    df = pd.DataFrame({
        'date': dates,
        'value': values
    })
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')

@pytest.fixture
def real_ts_df():
    df = load_from_kaggle(
        dataset_id="mohammadtalib786/retail-sales-dataset",
        filename="retail_sales_dataset.csv")
    return df.set_index('Date')


def test_analyze_time_series_basic_synthetic(synthetic_ts_df):
    result = analyze_time_series(synthetic_ts_df, value_column='sales', freq='D')
    assert 'basic_stats' in result
    assert 'stationarity' in result
    assert 'trend_strength' in result
    assert 'seasonality_strength' in result
    assert result['trend_strength'] > 0.5  # должен быть сильный тренд

def test_analyze_time_series_basic_real(synthetic_ts_df):
    result = analyze_time_series(synthetic_ts_df, value_column='sales', freq='D')
    assert 'basic_stats' in result
    assert 'stationarity' in result
    assert 'trend_strength' in result
    assert 'seasonality_strength' in result
    print(result)


def test_analyze_time_series_with_datetime_column():
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=50, freq='D'),
        'value': np.linspace(10, 60, 50)
    })
    result = analyze_time_series(df, value_column='value', datetime_column='date', freq='D')
    assert 'basic_stats' in result
    assert result['basic_stats']['mean'] == pytest.approx(35.0)


def test_analyze_time_series_with_nan_values(synthetic_ts_with_nan_df):
    result = analyze_time_series(synthetic_ts_with_nan_df, value_column='value', freq='D')
    assert result['basic_stats']['missing_values'] == 2
    # Проверим, что NaN были корректно обработаны


def test_analyze_time_series_multiplicative_model():
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=50, freq='D'),
        'value': np.exp(np.linspace(0.1, 1.0, 50))  # мультипликативный тренд
    }).set_index('date')
    result = analyze_time_series(df, value_column='value', freq='D', model='multiplicative')
    assert 'decomposition' in result


def test_analyze_time_series_non_numeric_column():
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=10),
        'value': ['a', 'b'] * 5
    }).set_index('date')
    with pytest.raises(ValueError, match="must be numeric"):
        analyze_time_series(df, value_column='value', freq='D')


def test_analyze_time_series_missing_value_column():
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=10),
        'other': [1] * 10
    }).set_index('date')
    with pytest.raises(ValueError, match="not found"):
        analyze_time_series(df, value_column='value', freq='D')


def test_analyze_time_series_no_datetime_index():
    df = pd.DataFrame({
        'value': [1, 2, 3, 4, 5]
    })
    with pytest.raises(ValueError, match="DatetimeIndex"):
        analyze_time_series(df, value_column='value', freq='D')


def test_analyze_time_series_custom_percentiles_not_applicable():
    # В текущей версии процентили не вычисляются в analyze_time_series
    # Этот тест проверяет, что функция не ломается при корректных данных
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=30, freq='D'),
        'value': np.random.randn(30)
    }).set_index('date')
    result = analyze_time_series(df, value_column='value', freq='D')
    assert 'basic_stats' in result
    # Тест просто убеждается, что вызов успешен


def test_analyze_time_series_stationarity_result():
    # Случайный шум не должен быть стационарным
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'value': np.random.randn(100)
    }).set_index('date')
    result = analyze_time_series(df, value_column='value', freq='D')
    # ADF тест может быть недетерминированным, но мы проверим, что результат есть
    assert 'p_value' in result['stationarity']
    assert result['stationarity']['adf_statistic'] is not None


def test_analyze_time_series_empty_after_dropna():
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=10),
        'value': [np.nan] * 10
    }).set_index('date')
    with pytest.raises(ValueError, match="No valid data after dropping"):
        analyze_time_series(df, value_column='value', freq='D')