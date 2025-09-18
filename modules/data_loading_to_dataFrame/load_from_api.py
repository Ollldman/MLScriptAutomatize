import pandas as pd
import logging
from typing import Any, Dict, Optional, Union
import os
from settings import settings

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import kagglehub
from kagglehub import KaggleDatasetAdapter
from datasets import load_dataset as hf_load_dataset
from sklearn import datasets as sklearn_datasets
import requests
from io import StringIO, BytesIO


import os


# ========================
# LOADER FUNCTIONS
# ========================

def load_from_kaggle(dataset_id: str, filename: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Загружает датасет с Kaggle через kagglehub с использованием PANDAS адаптера.

    :param dataset_id: Например, "username/dataset_name"
    :param filename: Имя файла внутри датасета (например, "winemag-data-130k-v2.csv"). 
    :param kwargs: Дополнительные аргументы, передаваемые в pd.read_csv (если используется fallback)
    :return: pd.DataFrame или None в случае ошибки
    """
    # Помещаем API kaggle в окружение если настройки заданы
    if not settings.KAGGLE_KEY and not settings.KAGGLE_KEY:
        logger.warning("Не заданы API credentials of Kaggle's public API")
        return None 
    else:
        os.environ["KAGGLE_USERNAME"] = settings.KAGGLE_USERNAME
        os.environ["KAGGLE_KEY"] = settings.KAGGLE_KEY
    try:
        # Скачиваем датасет
        logger.info(f"Загрузка датасета с Kaggle: {dataset_id}")
        data: pd.DataFrame = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            dataset_id,
            filename,
            pandas_kwargs=kwargs
        )
        return data

    except Exception as e:
        logger.error(f"Ошибка при загрузке с Kaggle: {e}")
        return None
    



def load_from_huggingface(dataset_id: str, config_name: Optional[str] = None,
                          split: str = "train", **kwargs) -> Optional[pd.DataFrame]:
    """
    Загружает датасет с Hugging Face Datasets.

    :param dataset_id: Например, "imdb", "glue", "rajpurkar/squad", и т.д.
    :param config_name: Конфиг (например, "mrpc" для glue)
    :param split: "train", "test", "validation"
    :return: pd.DataFrame или None
    """

    try:
        logger.info(f"🤗 Загрузка датасета с Hugging Face: {dataset_id}, split={split}")
        dataset = hf_load_dataset(dataset_id, config_name, split=split)
        df = dataset.to_pandas()
        logger.info(f"Загружено {len(df)} строк")
        return df
    except Exception as e:
        logger.error(f"Ошибка при загрузке с Hugging Face: {e}")
        return None


def load_from_uci(url: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Загружает датасет с UCI Machine Learning Repository.

    :param url: Прямая ссылка на .data или .csv файл
    :return: pd.DataFrame или None
    """
    try:
        logger.info(f"🌐 Загрузка с UCI: {url}")
        response = requests.get(url)
        response.raise_for_status()

        # Попробуем как CSV
        content = StringIO(response.text)
        df = pd.read_csv(content, **kwargs)
        logger.info(f"Загружено {len(df)} строк")
        return df
    except Exception as e:
        logger.error(f"Ошибка при загрузке с UCI: {e}")
        return None


def load_from_sklearn(name: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Загружает встроенный датасет из sklearn.datasets.

    :param name: Имя функции, например, "load_iris", "load_boston", "fetch_california_housing"
    :return: pd.DataFrame или None
    """
    if not SKLEARN_AVAILABLE:
        logger.error("sklearn не установлен.")
        return None

    try:
        if not hasattr(sklearn_datasets, name):
            logger.error(f"Датасет '{name}' не найден в sklearn.datasets")
            return None

        logger.info(f"Загрузка датасета из sklearn: {name}")
        loader = getattr(sklearn_datasets, name)
        data = loader()

        # Конвертируем в DataFrame
        if hasattr(data, 'frame') and data.frame is not None:
            df = data.frame
        else:
            df = pd.DataFrame(data.data, columns=data.feature_names)
            if hasattr(data, 'target'):
                target_name = 'target' if not hasattr(data, 'target_names') else 'target'
                df[target_name] = data.target

        logger.info(f"Загружено {len(df)} строк")
        return df
    except Exception as e:
        logger.error(f"Ошибка при загрузке из sklearn: {e}")
        return None


# ========================
# MAIN UNIVERSAL LOADER
# ========================

def load_dataset(source: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Универсальная функция загрузки датасетов.

    Поддерживаемые источники:
      - 'kaggle': dataset_id, filename
      - 'huggingface': dataset_id, config_name, split
      - 'uci': url
      - 'sklearn': name

    Примеры:
      load_dataset('kaggle', dataset_id='zynicide/wine-reviews', filename='winemag-data_first150k.csv')
      load_dataset('huggingface', dataset_id='imdb', split='test')
      load_dataset('uci', url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
      load_dataset('sklearn', name='load_iris')
    """
    loaders = {
        'kaggle': load_from_kaggle,
        'huggingface': load_from_huggingface,
        'uci': load_from_uci,
        'sklearn': load_from_sklearn,
    }

    if source not in loaders:
        logger.error(f"Неизвестный источник: {source}. Доступные: {list(loaders.keys())}")
        return None

    try:
        return loaders[source](**kwargs)
    except Exception as e:
        logger.error(f"Необработанная ошибка при загрузке из {source}: {e}")
        return None