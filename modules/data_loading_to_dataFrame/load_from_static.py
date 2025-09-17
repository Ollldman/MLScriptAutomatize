import pandas as pd
from pandas import DataFrame
from typing import BinaryIO, TextIO, Union, Any, Dict
from io import BytesIO, StringIO


def load_from_csv(
    file_or_path: Union[
        str,
        bytes,
        BytesIO, 
        StringIO, 
        BinaryIO, 
        TextIO,
        Any],
    encoding: str = 'utf-8',
    sep: str = ',',
    **kwargs
) -> DataFrame:
    """
    Load data from CSV format from various sources:
        - local file path (str)
        - uploaded file object (e.g., from Streamlit, FastAPI, Django)
        - BytesIO/StringIO buffer
        - raw bytes

    Args:
        file_or_path: path to file or file-like object or bytes
        encoding: encoding of the file (default: 'utf-8')
        sep: separator (default: ',')
        **kwargs: additional arguments for pd.read_csv

    Returns:
        pandas.DataFrame
    """
    try:
        if isinstance(file_or_path, str):
            # Это путь к файлу
            print(f'Load CSV file from path: {file_or_path}')
            return pd.read_csv(
                filepath_or_buffer=file_or_path,
                encoding=encoding,
                sep=sep,
                **kwargs)

        elif isinstance(file_or_path, bytes):
            # Это сырые байты (например, из загрузчика)
            print('Load CSV from bytes')
            buffer = BytesIO(file_or_path)
            return pd.read_csv(
                filepath_or_buffer=buffer, 
                encoding=encoding, 
                sep=sep, 
                **kwargs)

        elif hasattr(file_or_path, 'read'):
            # Это файлоподобный объект (например, UploadedFile, BytesIO, StringIO)
            print('Load CSV from file-like object')
            # Проверим, бинарный ли это поток
            if isinstance(file_or_path, BytesIO):
                return pd.read_csv(
                    filepath_or_buffer=file_or_path,
                    encoding=encoding, 
                    sep=sep, 
                    **kwargs)
            else:
                # Для StringIO или текстовых потоков
                return pd.read_csv(
                    file_or_path, # type: ignore
                    encoding=encoding,
                    sep=sep,
                    **kwargs)

        else:
            raise ValueError(f"Unsupported file_or_path type: {type(file_or_path)}")

    except Exception as e:
        raise RuntimeError(f"Failed to load CSV: {e}")


def load_from_excel(
    file_or_path: Union[str, bytes, BytesIO, Any],
    sheet_name: Union[str, int, list, None] = 0,
    **kwargs
) -> Union[DataFrame , Dict[Union[str, int], DataFrame]]:
    """
    Load data from Excel format from various sources:
        - local file path (str)
        - uploaded file object
        - BytesIO buffer
        - raw bytes

    Args:
        file_or_path: path to file or file-like object or bytes
        sheet_name: sheet name or index (default: 0)
        **kwargs: additional arguments for pd.read_excel

    Returns:
        pandas.DataFrame or dict of DataFrames (if sheet_name=None or list)
    """
    try:
        if isinstance(file_or_path, str):
            print(f'Load Excel file from path: {file_or_path}, sheet: {sheet_name}')
            return pd.read_excel(file_or_path, sheet_name=sheet_name, **kwargs)

        elif isinstance(file_or_path, bytes):
            print('Load Excel from bytes')
            buffer = BytesIO(file_or_path)
            return pd.read_excel(buffer, sheet_name=sheet_name, **kwargs)

        elif hasattr(file_or_path, 'read'):
            print('Load Excel from file-like object')
            # pd.read_excel умеет работать с BytesIO и подобными
            return pd.read_excel(file_or_path, sheet_name=sheet_name, **kwargs)

        else:
            raise ValueError(f"Unsupported file_or_path type: {type(file_or_path)}")

    except Exception as e:
        raise RuntimeError(f"Failed to load Excel: {e}")