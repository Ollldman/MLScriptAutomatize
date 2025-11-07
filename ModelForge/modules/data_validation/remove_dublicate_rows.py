import pandas as pd
from typing import Dict, Any, Hashable, Optional, List, Union
from pydantic import BaseModel, ConfigDict

cDataFrame = Union[Dict[str, List[Any]], Dict[Hashable, Any], None]
class DeduplicationResult(BaseModel):
    """
    Type structure for serialization.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    status: str  # "success" or "no_duplicates"
    cleaned_dataframe: cDataFrame = None  
    removed_duplicates: Optional[Dict[str, Dict[str, Any]]] = None 



def remove_duplicate_rows(df: pd.DataFrame) -> DeduplicationResult:
    """
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
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    duplicated_mask = df.duplicated(keep='first')

    if not duplicated_mask.any():
        return DeduplicationResult(status="no_duplicates")

    removed_rows = df[duplicated_mask]
    removed_dict = {
        str(idx): row.to_dict()
        for idx, row in removed_rows.iterrows()
    }

    cleaned_df = df.drop_duplicates(keep='first').reset_index(drop=True)
    cleaned_dict = cleaned_df.to_dict(orient='list')

    return DeduplicationResult(
        status="success",
        cleaned_dataframe=cleaned_dict,
        removed_duplicates=removed_dict
    )