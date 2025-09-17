import pandas as pd
from pandas import DataFrame
from typing import Union



def load_from_csv(file_path: str) -> DataFrame:
    """
    Load data from *.csv format files. 
    Attention! Data must be organized in a tabular form.
    Returns:
        pandas.DataFrame
    """
    print(f'Load CSV file from {file_path}')
    return pd.read_csv(file_path)
    

def load_from_excel(file_path: str, sheet_name: Union[str, int]) -> DataFrame:
    """
    Load data from *.csv format files. 
    Attention! Data must be organized in a tabular form.
    Returns:
        pandas.DataFrame
    """
    print(f'Load excel file from {file_path} and sheet: {sheet_name}')
    return pd.read_excel(file_path, sheet_name)