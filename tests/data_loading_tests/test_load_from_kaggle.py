from modules.data_loading_to_dataFrame.load_from_api import (
    load_from_kaggle,
    load_from_uci)
from pandas import DataFrame
from typing import Optional

TEST_DATASET_KAGGLE: tuple[str, str] = ("navjotkaushal/coffee-sales-dataset","Coffe_sales.csv")
TEST_DATASET_UCI: int = 186
WRONG_TEST_DATASET: tuple[str, str] = ("haha", "hoho")

def test_load_from_kaggle():
    df: Optional[DataFrame] = load_from_kaggle(TEST_DATASET_KAGGLE[0], TEST_DATASET_KAGGLE[1])
    assert df is not None
    print(df.head(2))

def test_load_from_kaggle_wrong_data():
    df: Optional[DataFrame] = load_from_kaggle(WRONG_TEST_DATASET[0], WRONG_TEST_DATASET[1])
    assert df is None


def test_load_from_uci():
    df: Optional[DataFrame] = load_from_uci(TEST_DATASET_UCI)
    assert df is not None
    print(df.head(2))