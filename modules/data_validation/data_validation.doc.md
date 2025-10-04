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