# test_normalization.py
import pytest
import pandas as pd
import numpy as np
from ModelForge.modules.data_validation.normalize_numerical_columns import normalize_numerical_columns


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1.0, 2.0, 3.0, 4.0],
        'B': [10, 20, 30, 40],
        'C': ['x', 'y', 'z', 'w']  # non-numeric
    })


def test_normalize_all_numeric_columns_standard(sample_df):
    result = normalize_numerical_columns(sample_df, method='standard')
    # StandardScaler: mean ≈ 0, std ≈ 1
    assert np.allclose(result['A'].mean(), 0, atol=1e-10)
    assert np.allclose(result['A'].std(ddof=0), 1, atol=1e-10)
    assert np.allclose(result['B'].mean(), 0, atol=1e-10)
    assert np.allclose(result['B'].std(ddof=0), 1, atol=1e-10)
    assert result['C'][0] == 'x' # non-numeric preserved


def test_normalize_specific_column_by_name_minmax(sample_df):
    result = normalize_numerical_columns(sample_df, column='A', method='minmax')
    assert result['A'].min() == 0.0
    assert result['A'].max() == 1.0
    assert 'B' in result.columns and not np.allclose(result['B'], sample_df['B']) is False
    # 'B' should remain unchanged
    pd.testing.assert_series_equal(result['B'], sample_df['B'])


def test_normalize_specific_column_by_index(sample_df):
    # Column 'B' is at index 1
    result = normalize_numerical_columns(sample_df, column=1, method='minmax')
    assert result['B'].min() == 0.0
    assert result['B'].max() == 1.0


def test_normalize_raises_value_error_on_non_numeric_column():
    df = pd.DataFrame({'cat': ['a', 'b', 'c']})
    with pytest.raises(ValueError, match="No numeric columns found"):
        normalize_numerical_columns(df)


def test_normalize_raises_value_error_on_invalid_method(sample_df):
    with pytest.raises(ValueError, match="Unsupported normalization method"):
        normalize_numerical_columns(sample_df, method='invalid_method')