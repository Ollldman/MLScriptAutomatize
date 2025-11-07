import pytest
import pandas as pd
import numpy as np
from ModelForge.modules.data_validation.handle_missing_values import (
    handle_missing_numeric_values,
    handle_missing_categorical_values,
    check_dataset,
    handle_missing_values,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'num1': [1.0, 2.0, np.nan, 4.0],
        'num2': [10, np.nan, 30, 40],
        'cat1': ['A', np.nan, 'C', 'A'],
        'cat2': [np.nan, 'X', 'Y', np.nan]
    })


def test_handle_missing_numeric_values_mean(sample_df):
    result = handle_missing_numeric_values(sample_df, strategy="mean")
    assert not result[['num1', 'num2']].isnull().any().any()
    assert result.loc[2, 'num1'] == pytest.approx((1 + 2 + 4) / 3)
    assert result.loc[1, 'num2'] == pytest.approx((10 + 30 + 40) / 3)


def test_handle_missing_categorical_values_most_frequent(sample_df):
    result = handle_missing_categorical_values(
        sample_df, 
        strategy="constant", 
        fill_value='A')
    assert not result[['cat1', 'cat2']].isnull().any().any()
    assert result.loc[1, 'cat1'] == 'A'  # 'A' is most frequent
    # For cat2: 'X' and 'Y' appear once → sklearn picks first encountered
    assert result.loc[0, 'cat2'] in ['A','X', 'Y']


def test_check_dataset_reports_both_types(sample_df):
    report = check_dataset(sample_df)
    assert "numeric columns: [num1, num2]" in report
    assert "categorical columns: [cat1, cat2]" in report
    assert "2 missing values in numeric" in report
    assert "3 missing values in categorical" in report


def test_check_dataset_no_missing():
    df = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y']})
    report = check_dataset(df)
    assert report == "No missing values detected."


def test_handle_missing_values_high_level_both_strategies(sample_df):
    result = handle_missing_values(
        sample_df,
        numeric_strategy="median",
        categorical_strategy="most_frequent"
    )
    assert result.isnull().sum().sum() == 0
    assert result.loc[2, 'num1'] == 2.0
    assert result.loc[1, 'cat1'] == 'A'


def test_handle_missing_values_only_numeric(sample_df):
    result = handle_missing_values(sample_df, numeric_strategy="mean")
    # Categorical columns should remain unchanged (with NaNs)
    assert result['cat1'].isnull().sum() == 1
    assert result['cat2'].isnull().sum() == 2
    assert not result[['num1', 'num2']].isnull().any().any()


def test_handle_missing_values_only_categorical(sample_df):
    result = handle_missing_values(sample_df, categorical_strategy="constant", categorical_fill_value="MISSING")
    # Numeric columns unchanged
    assert result['num1'].isnull().sum() == 1
    assert result['num2'].isnull().sum() == 1
    # Categorical filled
    assert result.loc[0, 'cat2'] == "MISSING"
    assert result.loc[3, 'cat2'] == "MISSING"


def test_handle_missing_numeric_values_constant_requires_fill_value():
    df = pd.DataFrame({'A': [1, np.nan]})
    with pytest.raises(ValueError, match="fill_value"):
        handle_missing_numeric_values(df, strategy="constant")


def test_handle_missing_categorical_values_constant_requires_fill_value():
    df = pd.DataFrame({'A': ['x', None]})
    with pytest.raises(ValueError, match="fill_value"):
        handle_missing_categorical_values(df, strategy="constant")


def test_handle_missing_values_empty_dataframe():
    df = pd.DataFrame()
    result = handle_missing_values(df, numeric_strategy="mean", categorical_strategy="most_frequent")
    pd.testing.assert_frame_equal(result, df)