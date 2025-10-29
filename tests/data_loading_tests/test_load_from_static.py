from typing import Union
from modules.data_loading_to_dataFrame import load_from_static
from settings import Settings
from pandas import DataFrame


def test_loading_data_from_dynamic_csv(app_settings: Settings):
    # Download dataset from Kaggle
    # Movies Dataset (TMDB) – Ratings, Popularity, Votes
    # https://www.kaggle.com/datasets/kajaldeore04/movies-dataset-tmdb-ratings-popularity-votes
    # and run pytest -s -v

    data: DataFrame = load_from_static.load_from_csv(
        app_settings.TEST_DATASETS_PATH+"diabetes.csv")
    
    print(data.head(2))


def test_loading_from_dynamic_excel(app_settings: Settings):
    # Same dataset, only in excel format file
    data: Union[DataFrame, dict] = load_from_static.load_from_excel(
        app_settings.TEST_DATASETS_PATH+"diabetes.xlsx"
    )
    if isinstance(data, DataFrame):
        print(data.head(2))
    else:
        print(f'pass for {type(data)}')