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
## 5. feature_to_datetime