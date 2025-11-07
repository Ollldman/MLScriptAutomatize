# test_datetime_conversion.py
import pytest
import pandas as pd
from ModelForge.modules.data_validation.feature_to_datetime import convert_column_to_datetime


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'date_str': ['2023-01-01', '2023-02-15', '2023/03/10', 'invalid'],
        'value': [1, 2, 3, 4]
    })


def test_convert_column_to_datetime_by_name(sample_df):
    result = convert_column_to_datetime(sample_df, 'date_str', errors='coerce')
    assert pd.api.types.is_datetime64_any_dtype(result['date_str'])
    assert pd.isna(result.loc[3, 'date_str'])  # 'invalid' → NaT


def test_convert_column_to_datetime_by_index(sample_df):
    # 'date_str' is at index 0
    result = convert_column_to_datetime(sample_df, 0, errors='coerce')
    assert pd.api.types.is_datetime64_any_dtype(result['date_str'])


def test_convert_column_to_datetime_raise_on_invalid(sample_df):
    with pytest.raises(Exception):
        convert_column_to_datetime(sample_df, 'date_str', errors='raise')


def test_convert_column_to_datetime_invalid_column_name(sample_df):
    with pytest.raises(ValueError, match="does not exist"):
        convert_column_to_datetime(sample_df, 'nonexistent_col', errors='coerce')


def test_convert_column_to_datetime_invalid_input_type():
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        convert_column_to_datetime("not_a_dataframe", 'col')