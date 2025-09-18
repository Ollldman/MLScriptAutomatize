from settings import settings
import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional, Dict, Any
import logging
import sqlparse


#Настройка логирования:
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class SQLRunner:
    """
    Universal class for execute SQL-based requests to 
    local DB and return pandas.DataFrame only
    from SELECT instructions.

    Attention - only SELECT instructions is available!
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
        """Создаёт движок SQLAlchemy."""
        try:
            self.engine = create_engine(self.connection_string, echo=False)
            logger.info("Движок SQLAlchemy создан.")
        except Exception as e:
            logger.error(f"Ошибка при создании движка: {e}")
            self.engine = None

    def test_connection(self) -> bool:
        """Проверяет, можно ли подключиться к БД."""
        if not self.engine:
            logger.error("Движок не создан.")
            return False
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Подключение к БД успешно.")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Ошибка подключения к БД: {e}")
            return False
        except Exception as e:
            logger.error(f"Неизвестная ошибка: {e}")
            return False
        
    def _is_select_query(self, sql: str) -> bool:
        """
        Проверяет, является ли запрос SELECT-запросом.

        :param sql: SQL-запрос
        :return: True, если это SELECT
        """
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return False

            first_statement = parsed[0]
            # Получаем тип первого токена верхнего уровня
            stmt_type = first_statement.get_type()
            return stmt_type.upper() == 'SELECT'
        except Exception as e:
            logger.error(f"Ошибка при разборе SQL: {e}")
            return False

    def query_to_dataframe(
            self,
            sql: str,
            params: Optional[Dict[str, Any]] = None
    ) -> DataFrame | None:
        """
        Выполняет SQL-запрос и возвращает результат как pandas.DataFrame.
        Разрешены ТОЛЬКО SELECT-запросы.

        :param sql: SQL-запрос
        :param params: Параметры для подстановки
        :return: pd.DataFrame или None в случае ошибки
        """
        if not self.engine:
            logger.error("Движок не инициализирован.")
            return None

        # !!Блокировка не-SELECT запросов
        if not self._is_select_query(sql):
            logger.error("Запрос отклонён: разрешены только SELECT-запросы.")
            return None

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(sql), conn, params=params)
                logger.info(f"SELECT-запрос выполнен. Получено строк: {len(df)}")
                return df
        except SQLAlchemyError as e:
            logger.error(f"Ошибка выполнения SQL-запроса: {e}")
            return None
        except Exception as e:
            logger.error(f"Неизвестная ошибка при выполнении запроса: {e}")
            return None

    def close(self):
        """Закрывает движок."""
        if self.engine:
            self.engine.dispose()
            logger.info("Движок соединения закрыт.")



def sql_to_df(
        connection_string:str,
        sql:str, 
        params: Optional[Dict[str, Any]] = None) -> Optional[DataFrame]:
    """
    Удобная функция для быстрого выполнения SQL → pandas.DataFrame.
    Создаёт экземпляр SQLExecutor, проверяет подключение и выполняет запрос.
    Разрешены только SELECT-запросы.
    """
    executor = SQLRunner(connection_string)
    if not executor.test_connection():
        return None
    return executor.query_to_dataframe(sql, params)

DB = SQLRunner