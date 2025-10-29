import asyncio
from settings import Settings
from modules.data_loading_to_dataFrame.acync_load_from_db import (
    async_sql_to_df,
    AsyncSQLRunner,
    AsyncDB)
from pandas import DataFrame
import numpy as np
from typing import AsyncGenerator,Optional
import pytest
import pytest_asyncio
import time
"""
Using demo db from https://postgrespro.ru/education/demodb

Source db - postgresql on localhost via .env connection string config

Tests: 
6 SQL_QUERY and etc.

1. Test connection status +
2. Test SELECT-requests
3. Test not SELECT requests
4. Test warning requests
etc.
"""
SQL_QUERY_1: str = "SELECT * FROM airplanes_data LIMIT 2;"
SQL_QUERY_2: str = "SELECT * FROM bookings LIMIT :x;"
SQL_QUERY_3: str = "SELECT * FROM airports_data LIMIT 2;"
SQL_QUERY_4: str = "ALTER TABLE airplanes_data ADD COLUMN construction text;"
SQL_QUERY_5: str = "DELETE FROM airplanes_data WHERE airplane_code = '339';"
SQL_QUERY_6: str = "This is not a SQL!!!"


QUERIES = [
    "SELECT * FROM seats LIMIT 10;",
    "SELECT * FROM airports_data LIMIT 10;",
    "SELECT * FROM bookings LIMIT 10;",
    "SELECT * FROM tickets LIMIT 10;",
    "SELECT * FROM flights LIMIT 10;",
]


@pytest.mark.asyncio
async def test_connection_status(app_settings: Settings):
    db: AsyncDB = AsyncSQLRunner(app_settings.AIO_DB_CONNECTION)
    if await db.test_connection():
        print('Connection!!')
        assert True
    else:
        print('Failed to connect!!')
        assert False


@pytest_asyncio.fixture(scope="function")
async def executor(app_settings: Settings)-> AsyncGenerator[AsyncSQLRunner, None]:
    ex: AsyncSQLRunner = AsyncSQLRunner(app_settings.AIO_DB_CONNECTION)
    assert await ex.test_connection(), "Не удалось подключиться к тестовой БД"
    yield ex
    await ex.close()

@pytest.mark.asyncio
async def test_select_query_works(executor: AsyncSQLRunner):
    df = await executor.query_to_dataframe(SQL_QUERY_1)
    if df:
        assert df is not np.nan
        assert not df.empty
        assert "model" in df.columns

@pytest.mark.asyncio
async def test_parametrized_select_works(executor: AsyncSQLRunner):
    df = await executor.query_to_dataframe(
        SQL_QUERY_2,
        {"x": 4}
    )
    assert df is not None
    assert len(df) == 4

@pytest.mark.asyncio
async def test_insert_rejected(executor: AsyncSQLRunner):
    df = await executor.query_to_dataframe(SQL_QUERY_3)
    assert df is not None 

# Not-SELECT requests:

@pytest.mark.asyncio
async def test_alter_rejected(executor: AsyncSQLRunner):
    df = await executor.query_to_dataframe(SQL_QUERY_4)
    assert df is None


@pytest.mark.asyncio
async def test_delete_rejected(executor: AsyncSQLRunner):
    df = await executor.query_to_dataframe(SQL_QUERY_5)
    assert df is None
    # check
    df = await executor.query_to_dataframe(
        "SELECT * FROM airplanes_data WHERE airplane_code = '789';")
    assert df is not None

@pytest.mark.asyncio
async def test_not_query_rejected(executor: AsyncSQLRunner):
    df = await executor.query_to_dataframe(SQL_QUERY_6)
    assert df is None

@pytest.mark.asyncio
async def test_multiple_statements_rejected(executor: AsyncSQLRunner):
    df = await executor.query_to_dataframe("SELECT 1; DROP TABLE employees;")
    assert df is None

@pytest.mark.asyncio
async def test_sql_to_df_function_select_only(app_settings: Settings):
    df =await async_sql_to_df(
        app_settings.AIO_DB_CONNECTION, 
        SQL_QUERY_1)
    assert df is not None
    assert df.iloc[0]['range'] == 6500

    # Проверка блокировки через функцию
    df2 = await async_sql_to_df(
        app_settings.AIO_DB_CONNECTION, 
        "DELETE FROM airports")
    assert df2 is None

@pytest.mark.asyncio
async def test_multiple_queries_async(executor: AsyncSQLRunner):
    """Замер времени выполнения 5 запросов ПАРАЛЛЕЛЬНО (асинхронно)."""

    # Синхронная обёртка для benchmark
    async def _run()-> list[Optional[DataFrame]]:
        tasks = [
            executor.query_to_dataframe(sql)
            for sql in QUERIES
        ]
        return await asyncio.gather(*tasks)
    start: float = time.perf_counter()
    dfs: list[Optional[DataFrame]] = await _run()
    elapsed: float = time.perf_counter() - start
    print(f"\n[ASYNC BENCHMARK] 5 параллельных запросов: {elapsed:.4f} секунд")

    assert len(dfs) == len(QUERIES)
    for df in dfs:
        assert isinstance(df, DataFrame)
        assert not df.empty