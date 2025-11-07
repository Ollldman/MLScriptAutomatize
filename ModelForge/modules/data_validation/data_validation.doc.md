# Документация по работе модулей валидации данных, загруженных из различных источников.

## 1. remove_dublicate_rows.py:

В модуле реализован BaseModel типизатор DublicationResult для возможной сериализации результата удаления дубликатов из pandas.DataFrame.

- `class DeduplicationResult(BaseModel)`:
    """
    Type structure for serialization.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    status: str  # "success" or "no_duplicates"
    cleaned_dataframe: cDataFrame = None  
    removed_duplicates: Optional[Dict[str, Dict[str, Any]]] = None

- `def remove_duplicate_rows`
Removes duplicate rows from a pandas DataFrame and returns a structured result.

    
    Args:
    ----------

    df : pd.DataFrame
        Input DataFrame.

        
    Returns:
    ----------

    DeduplicationResult
        An object with the following fields:
        - status: "no_duplicates" or "success"
        - cleaned_dataframe: DataFrame without duplicates in dict format (orient='list'), or None
        - removed_duplicates: dictionary {str(original_index): {column: value}}, or None
    
## 2. handle_missing_values.py:

В модуле реализованы 4 функции, основная задача которых - разобраться в отсутствующих значениях полей в предоставленном pandas.DataFrame. Т.к. мы можем иметь дело с числовыми и категориальными признаками или просто строками-объектами, наша задача принять универсальное решение для отсутвующих значений этих типов.
В первую очередь, проверим наличие отсутствующих значений:

- `def check_dataset`
Analyze a pandas DataFrame and generate a diagnostic report about missing values,
distinguishing between numeric and categorical columns.

Returns a human-readable string with counts and column names containing missing values.

Args:
    df (pd.DataFrame): Input DataFrame to analyze.

Returns:
    str: Diagnostic message describing missing data distribution.

Добавлена универсальная функция, которая должна применяться на основании доклада от первой функции. Тут мы и принимаем решение о том, как обработать полученные значения:

- `def handle_missing_values`
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

Дополнительно в функции производиться итоговый расчет обработанных значений в логирование.

## 3. normilize_numerical_columns.py

Модуль предоставляет функцию универсальной нормализации данных в колонке или на всех числовых значениях в pandas.DataFrame. На выбор предоставляется два метода - Standard, MinMax.

- `def normalize_numerical_columns`
Normalize numerical columns in a pandas DataFrame using StandardScaler or MinMaxScaler.

    If `column` is not provided, all numeric columns are normalized.
    If `column` is provided, only that column is normalized (must be numeric).

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str or int, optional): Specific column to normalize.
            If None, all numeric columns are processed.
        method (str): Normalization method. Must be 'standard' (z-score) or 'minmax'. Default is 'standard'.
        inplace (bool): If True, modifies the input DataFrame. Otherwise, returns a copy. Default is False.

    Returns:
        pd.DataFrame: DataFrame with normalized numerical columns.

    Raises:
        TypeError: If input is not a pandas DataFrame.
        ValueError: If specified column does not exist, is not numeric, or if no numeric columns are found.

## 4. feature_encoding
Модуль отвечает за обработку переданного объекта pandas.DataFrame применяя кодирование категориальных и строковых данных.
### Key Features

| Feature | Description |
|--------|-------------|
| **Input validation** | Checks DataFrame type, column existence, and data type |
| **Flexible input** | Accepts column by name (`str`) or index (`int`) |
| **Logging** | Every step is logged via Python’s `logging` module |
| **Sklearn-based** | Uses `OneHotEncoder` and `LabelEncoder` for compatibility with ML pipelines |
| **NaN handling** | Converts `NaN` to `"NaN"` string to avoid encoder errors |
| **Extensible** | Easy to add more encoders (e.g., `OrdinalEncoder`) |

---

### Example Usage

```python
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

df = pd.DataFrame({
    'color': ['red', 'blue', None, 'red'],
    'size': ['M', 'L', 'M', 'S']
})

# One-hot encode 'color'
df_ohe = encode_categorical_column(df, 'color', method='one_hot')

# Label encode 'size'
df_le = encode_categorical_column(df, 'size', method='label')
```
- `def encode_categorical_column`(
    df: pd.DataFrame,
    column: Union[str, int],
    method: str = "one_hot",
    **kwargs,
) -> pd.DataFrame:

High-level function to encode a categorical column using the specified method.

Validates that:
    - Input is a pandas DataFrame
    - Column exists
    - Column is of a categorical-like type (object or category)

Args:
    df (pd.DataFrame): Input DataFrame.
    column (str or int): Column name or index.
    method (str): Encoding method. Must be 'one_hot' or 'label'. Default is 'one_hot'.
    **kwargs: Additional arguments passed to the underlying encoder function.

Returns:
    pd.DataFrame: Encoded DataFrame.

Raises:
    TypeError: If df is not a pandas DataFrame.
    ValueError: If column is invalid or method is unsupported.
## 5. feature_to_datetime
Модуль отвечает за обработку переданного объекта pandas.DataFrame в верный форматы времени и даты.
### Key Features

| Feature | Description |
|--------|-------------|
| **Flexible column input** | Accepts `str` (name) or `int` (position) |
| **Safe by default** | Returns a copy unless `inplace=True` |
| **Error control** | Supports `'raise'`, `'coerce'`, `'ignore'` |
| **Format support** | Optional `format` for performance and precision |
| **Logging** | Logs conversion success/failure with context |
| **Type safety** | Full `typing` annotations |
| **Sklearn/pandas compatible** | Uses standard `pandas.to_datetime` |

---

### Example Usage

```python
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

df = pd.DataFrame({
    'date': ['2023-01-01', '2023-02-15', 'invalid'],
    'value': [1, 2, 3]
})

# Coerce invalid dates to NaT
df_clean = convert_column_to_datetime(df, 'date', errors='coerce')
print(df_clean.dtypes)
```
- `def convert_column_to_datetime`(
    df: pd.DataFrame,
    column: Union[str, int],
    format: Optional[str] = None,
    errors: str = "raise",
    inplace: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:

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

## 6. handle_outliers
- `def handle_outliers`(
    df: pd.DataFrame,
    column: Union[str, int],
    method: Literal["iqr", "zscore"] = "iqr",
    handling: Literal["remove", "clip", "impute"] = "clip",
    impute_strategy: Literal["mean", "median"] = "median",
    threshold: float = 1.5,
    z_threshold: float = 3.0,
    inplace: bool = False,
) -> pd.DataFrame:

Detect and handle outliers in a numeric column of a pandas DataFrame.

Supported detection methods:

- 'iqr': Uses Interquartile Range (Q1 - threshold*IQR, Q3 + threshold*IQR)
- 'zscore': Uses Z-score (|z| > z_threshold)

Supported handling strategies:

- 'remove': Drop rows with outliers
- 'clip': Cap outliers to the nearest non-outlier boundary
- 'impute': Replace outliers with mean or median of non-outlier values

Args:

df (pd.DataFrame): Input DataFrame.

column (str or int): Column name or index to process.

method (str): Outlier detection method. Default is 'iqr'.

handling (str): How to handle detected outliers. Default is 'clip'.

impute_strategy (str): Strategy for imputation ('mean' or 'median'). Used only if handling='impute'.

threshold (float): IQR multiplier (e.g., 1.5). Used only if method='iqr'.

z_threshold (float): Z-score cutoff (e.g., 3.0). Used only if method='zscore'.

inplace (bool): If True, modifies the input DataFrame. Otherwise, returns a copy.

Returns:

pd.DataFrame: DataFrame with outliers handled as specified.

Raises:

TypeError: If input is not a pandas DataFrame.

ValueError: If column is not numeric or does not exist.