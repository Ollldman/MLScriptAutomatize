import pandas as pd


def data_statistics(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe()
