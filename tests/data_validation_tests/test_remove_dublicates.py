import pandas as pd
import numpy as np
import pytest

from modules.data_validation.remove_dublicate_rows import (
    remove_duplicate_rows,
    DeduplicationResult)



class TestRemoveDuplicateRows:

    def test_no_duplicates(self):
        """Test when there are no duplicate rows."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })
        result = remove_duplicate_rows(df)

        assert isinstance(result, DeduplicationResult)
        assert result.status == "no_duplicates"
        assert result.cleaned_dataframe is None
        assert result.removed_duplicates is None

    def test_with_duplicates(self):
        """Test removal of duplicate rows (keep first)."""
        df = pd.DataFrame({
            'A': [1, 2, 2, 3, 2],
            'B': ['x', 'y', 'y', 'z', 'y']
        })

        result = remove_duplicate_rows(df)

        assert result.status == "success"
        assert result.cleaned_dataframe is not None
        assert result.removed_duplicates is not None

        # Check cleaned DataFrame
        expected_cleaned = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        }).to_dict(orient='list')
        assert result.cleaned_dataframe == expected_cleaned

        # Check removed duplicates: indices 2 and 4 should be removed
        expected_removed = {
            '2': {'A': 2, 'B': 'y'},
            '4': {'A': 2, 'B': 'y'}
        }
        assert result.removed_duplicates == expected_removed

    def test_all_rows_duplicate(self):
        """Test when all rows are identical."""
        df = pd.DataFrame({
            'X': [5, 5, 5],
            'Y': ['same', 'same', 'same']
        })

        result = remove_duplicate_rows(df)

        assert result.status == "success"
        # Only first row remains
        assert result.cleaned_dataframe == {'X': [5], 'Y': ['same']}
        assert result.removed_duplicates == {
            '1': {'X': 5, 'Y': 'same'},
            '2': {'X': 5, 'Y': 'same'}
        }

    def test_empty_dataframe(self):
        """Test with an empty DataFrame."""
        df = pd.DataFrame()
        result = remove_duplicate_rows(df)

        assert result.status == "no_duplicates"
        assert result.cleaned_dataframe is None
        assert result.removed_duplicates is None

    def test_single_row(self):
        """Test with a single row (no duplicates possible)."""
        df = pd.DataFrame({'col': [42]})
        result = remove_duplicate_rows(df)

        assert result.status == "no_duplicates"

    def test_with_nan_values(self):
        """Test that NaN values are handled correctly (pandas treats NaN==NaN as False, but duplicated() works as expected)."""
        df = pd.DataFrame({
            'A': [1.0, np.nan, 1.0, np.nan],
            'B': ['a', 'b', 'a', 'b']
        })

        result = remove_duplicate_rows(df)

        assert result.status == "success"
        # Rows 0 and 2 are duplicates; rows 1 and 3 are duplicates
        assert len(result.cleaned_dataframe['A']) == 2 #type:ignore

        expected_removed_df = pd.DataFrame({
        'A': [1.0, np.nan],
        'B': ['a', 'b']
        }, index=['2', '3'])
        actual_removed_df = pd.DataFrame.from_dict(result.removed_duplicates, # type:ignore
                                                    orient='index').astype(object)# type:ignore
        
        pd.testing.assert_frame_equal(
        actual_removed_df,
        expected_removed_df,
        check_dtype=False,
        check_index_type=False)

    def test_non_integer_index(self):
        """Test with string index — should still work, and keys in removed_duplicates must be strings."""
        df = pd.DataFrame({
            'val': [10, 20, 10]
        }, index=['row1', 'row2', 'row3'])

        result = remove_duplicate_rows(df)

        assert result.status == "success"
        assert 'row3' in result.removed_duplicates # type:ignore
        assert result.removed_duplicates['row3'] == {'val': 10} # type:ignore

    def test_invalid_input_type(self):
        """Test that TypeError is raised for non-DataFrame input."""
        with pytest.raises(TypeError, match="Input must be a pandas DataFrame."):
            remove_duplicate_rows("not a dataframe") # type:ignore

        with pytest.raises(TypeError, match="Input must be a pandas DataFrame."):
            remove_duplicate_rows(None) # type:ignore

        with pytest.raises(TypeError, match="Input must be a pandas DataFrame."):
            remove_duplicate_rows([1, 2, 3]) # type:ignore