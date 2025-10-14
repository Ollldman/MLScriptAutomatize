import logging
import sqlparse
import pandas as pd
import numpy as np
from pandas import DataFrame
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional, Dict, Any



#Настройка логирования:
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class AsyncSQLRunner:
    """
    Universal async class for execute SQL-based requests to 
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
        self.engine: Optional[AsyncEngine] = None
        self._create_engine()

    def _create_engine(self) -> None:
        """This function creates an asynchronous engine SQLAlchemy."""
        try:
            self.engine = create_async_engine(self.connection_string, echo=False)
            logger.info("The engine SQLAlchemy was created!")
        except Exception as e:
            logger.error(f"Error when creating an engine: {e}")
            self.engine = None

    async def test_connection(self) -> bool:
        """Check connection to DB."""
        if not self.engine:
            logger.error("The engine is not created.")
            return False
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("Database connection successfully.")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Database connection error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unknown error: {e}")
            return False
        
    def _is_select_query(self, sql: str) -> bool:
        """
        This function checks whether the request is the SELECT-request.

        :param sql: SQL-request
        :return: True, if SELECT-request
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
            logger.error(f"Error with parse your SQL-request: {e}")
            return False

    async def query_to_dataframe(
            self,
            sql: str,
            params: Optional[Dict[str, Any]] = None
    ) -> DataFrame | None:
        """
        Performs SQL request and returns the result as Pandas.Dataframe.
        Only SELECT is allowed.

        :param sql: SQL-request
        :param params: params to connection.execute(...)
        :return: pd.DataFrame or None 
        """
        if not self.engine:
            logger.error("The engine is not initialized.")
            return None

        # !!Блокировка не-SELECT запросов
        if not self._is_select_query(sql):
            logger.error("The request is rejected: only Select-requests are allowed.")
            return None

        try:
            async with self.engine.connect() as conn:
                output_from_db = await conn.execute(text(sql), params or {})
                rows = output_from_db.fetchall()
                columns = output_from_db.keys()

                df: DataFrame = pd.DataFrame(rows, columns=columns) #type:ignore
                df = df.replace({None: np.nan})
                logger.info(f"SELECT-request is complete. Rows: {len(df)}")
                return df
        except SQLAlchemyError as e:
            logger.error(f"Request error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unknown request error: {e}")
            return None

    async def close(self):
        """Close the engine."""
        if self.engine:
            await self.engine.dispose()
            logger.info("The engine is close.")



async def async_sql_to_df(
        connection_string:str,
        sql:str, 
        params: Optional[Dict[str, Any]] = None) -> Optional[DataFrame]:
    """
    This function provides quick performance SQL → Pandas.Dataframe.
    Creates an ASYNCSQLEXECUTOR copy, checks the connection and executes a request.
    Only Select-checks are allowed.
    """
    executor = AsyncSQLRunner(connection_string)
    if not await executor.test_connection():
        await executor.close()
        return None
    df: Optional[DataFrame] = await executor.query_to_dataframe(sql, params)
    await executor.close()
    
    return df

AsyncDB = AsyncSQLRunner