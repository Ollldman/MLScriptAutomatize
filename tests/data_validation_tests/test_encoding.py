import pytest
import pandas as pd
import numpy as np
from ModelForge.modules.data_validation.feature_encoding import encode_categorical_column, apply_one_hot_encoding, apply_label_encoding


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'color': ['red', 'blue', 'brown', 'red'],
        'size': ['M', 'L', 'M', 'S'],
        'price': [10.0, 20.0, 30.0, 40.0],  # numeric column
    })


def test_encode_categorical_column_one_hot(sample_df):
    result = encode_categorical_column(sample_df, 'color', method='one_hot')
    assert 'color_red' in result.columns
    assert 'color_blue' in result.columns
    assert 'color_brown' in result.columns
    assert 'color' not in result.columns
    assert result.loc[2, 'color_brown'] == 1.0


def test_encode_categorical_column_label(sample_df):
    result = encode_categorical_column(sample_df, 'size', method='label')
    assert 'size_encoded' in result.columns
    assert 'size' not in result.columns
    # 'L', 'M', 'S' → 3 unique → labels 0,1,2 (order may vary, but all non-null)
    assert not result['size_encoded'].isnull().any()


def test_encode_categorical_column_invalid_method(sample_df):
    with pytest.raises(ValueError, match="Unsupported encoding method"):
        encode_categorical_column(sample_df, 'color', method='invalid')


def test_encode_categorical_column_column_not_found(sample_df):
    with pytest.raises(ValueError, match="not exist"):
        encode_categorical_column(sample_df, 'nonexistent', method='one_hot')


def test_encode_categorical_column_non_dataframe_input():
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        encode_categorical_column("not a dataframe", 'col')


def test_encode_categorical_column_by_index(sample_df):
    # 'color' is at index 0
    result = encode_categorical_column(sample_df, 0, method='label')
    assert 'color_encoded' in result.columns


def test_encode_categorical_column_warns_on_numeric_column(caplog, sample_df):
    with caplog.at_level("WARNING"):
        encode_categorical_column(sample_df, 'price', method='label')
    assert "not categorical" in caplog.text


def test_apply_one_hot_encoding_keeps_other_columns(sample_df):
    result = apply_one_hot_encoding(sample_df, 'color')
    assert 'price' in result.columns
    assert 'size' in result.columns


def test_apply_label_encoding_with_drop_original_false(sample_df):
    result = apply_label_encoding(sample_df, 'size', drop_original=False)
    assert 'size' in result.columns
    assert 'size_encoded' in result.columns


def test_encode_categorical_column_empty_category():
    df = pd.DataFrame({'cat': [np.nan, np.nan, np.nan]})
    result = encode_categorical_column(df, 'cat', method='one_hot')
    assert 'cat_nan' in result.columns
    assert result['cat_nan'].sum() == 3
    print(result)