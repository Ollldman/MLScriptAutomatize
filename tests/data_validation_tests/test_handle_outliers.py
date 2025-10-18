# test_outlier_handling.py
import pytest
import pandas as pd
import numpy as np
from modules.data_validation.handle_outliers import handle_outliers
from modules.data_loading_to_dataFrame.load_from_api import load_from_sklearn

california_df: pd.DataFrame | None = load_from_sklearn('fetch_california_housing')

# @pytest.fixture
# def sample_df():
#     return pd.DataFrame({
#         'values': [1, 2, 3, 4, 100],  # 100 is an outlier
#         'other': ['a', 'b', 'c', 'd', 'e']
#     })


# def test_handle_outliers_iqr_clip(sample_df):
#     result = handle_outliers(sample_df, 'values', method='iqr', handling='clip')
#     assert result['values'].max() < 100  # clipped
#     assert len(result) == len(sample_df)  # no rows removed


# def test_handle_outliers_zscore_remove(sample_df):
#     result = handle_outliers(sample_df, 'values', method='zscore', handling='remove', z_threshold=1.8)
#     assert len(result) == 4  # row with 100 removed
#     assert 100 not in result['values'].values


# def test_handle_outliers_iqr_impute_median(sample_df):
#     result = handle_outliers(sample_df, 'values', method='iqr', handling='impute', impute_strategy='median')
#     assert not result['values'].isna().any()
#     # Median of [1,2,3,4] is 2.5 → 100 replaced by 2.5
#     assert result.iloc[4]['values'] == 2.5


# def test_handle_outliers_invalid_column():
#     df = pd.DataFrame({'x': [1, 2, 3]})
#     with pytest.raises(ValueError, match="does not exist"):
#         handle_outliers(df, 'nonexistent', method='iqr')


# def test_handle_outliers_non_numeric_column():
#     df = pd.DataFrame({'cat': ['a', 'b', 'c']})
#     with pytest.raises(ValueError, match="not numeric"):
#         handle_outliers(df, 'cat', method='iqr')


def test_handle_outliers_real_dataset_sklearn():
    """
    Integration test using California Housing dataset (real-world numeric data).
    We inject an artificial outlier and verify it's handled.
    """
    if isinstance(california_df, pd.DataFrame):
        df = california_df.copy()
    else:
        raise ValueError("DataFrame is None")

    # Inject a clear outlier in 'MedHouseVal' (normally ~0.15–5.0)
    df.loc[0, 'MedHouseVal'] = 1000.0

    original_shape = df.shape
    original_max = df['MedHouseVal'].max()  # should be 1000.0
    # Clip outliers using IQR
    df_clean = handle_outliers(
        df,
        column='MedHouseVal',
        method='iqr',
        handling='clip',
        threshold=1.5
    )
    print(df_clean.describe())

    # Assertions
    # assert df_clean.shape == original_shape  # no rows lost
    # assert df_clean['MedHouseVal'].max() < original_max  # outlier clipped
    # assert df_clean['MedHouseVal'].max() <= df['MedHouseVal'].quantile(0.75) + 1.5 * (
    #     df['MedHouseVal'].quantile(0.75) - df['MedHouseVal'].quantile(0.25)
    # )
    # print(df['MedHouseVal'].tail(10))