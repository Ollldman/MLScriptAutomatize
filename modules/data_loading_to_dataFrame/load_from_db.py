from settings import settings
import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional, Dict, Any
import logging


#Настройка логирования:
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class SQLRunner:
    """
    Universal class for execute SQL-based requests to 
    local DB and return pandas.DataFrame only
    from SELECT instructions.
    """

    def __init__(
            self,
            connection_string: str
    ):
        """
        Connection to DB.

        :param connection_string: SQLAlchemy-like connection string
        (for example: 'postgresql://user:pass@host/db')
        """
        self.connection_string: str = connection_string
        self.engine = None
        self._create_engine()

    def _create_engine(self):
        pass

    def test_connection(self):
        pass

    def query_to_dataframe(
            self,
            sql: str,
            params: Optional[Dict[str, Any]] = None
    ) -> DataFrame | None:
        pass

    def close(self):
        pass



def sql_to_df(
        connection_string:str,
        sql:str, 
        params: Optional[Dict[str, Any]] = None) -> Optional[DataFrame]:
    pass