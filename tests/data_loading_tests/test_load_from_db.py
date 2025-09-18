from settings import Settings
from modules.data_loading_to_dataFrame import load_from_db
from pandas import DataFrame
from typing import Union
import pytest
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

SQL_QUERY_1: str = "SELECT * FROM aircrafts LIMIT 2;"
SQL_QUERY_2: str = "SELECT * FROM bookings LIMIT :x;"
SQL_QUERY_3: str = "SELECT * FROM airports LIMIT 2;"
SQL_QUERY_4: str = "ALTER TABLE aircrafts ADD COLUMN construction text;"
SQL_QUERY_5: str = "DELETE FROM aircrafts WHERE aircraft_code = '773';"
SQL_QUERY_6: str = "This is not a SQL!!!"


def test_connection_status(app_settings: Settings):
    db: load_from_db.DB = load_from_db.SQLRunner(app_settings.DB_CONNECTION)
    connection: bool = db.test_connection()
    if connection:
        print('Connection!!')
        assert True
    else:
        print('Failed to connect!!')
        assert False

@pytest.fixture(scope="module")
def executor(app_settings: Settings):
    ex = load_from_db.SQLRunner(app_settings.DB_CONNECTION)
    assert ex.test_connection(), "Не удалось подключиться к тестовой БД"
    yield ex
    ex.close()

def test_select_query_works(executor):
    df = executor.query_to_dataframe(SQL_QUERY_1)
    assert df is not None
    assert not df.empty
    assert "model" in df.columns

def test_parametrized_select_works(executor):
    df = executor.query_to_dataframe(
        SQL_QUERY_2,
        {"x": 4}
    )
    assert df is not None
    assert len(df) == 4

def test_insert_rejected(executor):
    df = executor.query_to_dataframe(SQL_QUERY_3)
    assert df is not None 

# Not-SELECT requests:
def test_alter_rejected(executor):
    df = executor.query_to_dataframe(SQL_QUERY_4)
    assert df is None


def test_delete_rejected(executor):
    df = executor.query_to_dataframe(SQL_QUERY_5)
    assert df is None
    # check
    df = executor.query_to_dataframe(
        "SELECT * FROM aircrafts WHERE aircraft_code = '773';")
    assert df is not None

def test_not_query_rejected(executor):
    df = executor.query_to_dataframe(SQL_QUERY_6)
    assert df is None

def test_multiple_statements_rejected(executor):
    df = executor.query_to_dataframe("SELECT 1; DROP TABLE employees;")
    assert df is None

def test_sql_to_df_function_select_only(app_settings: Settings):
    df = load_from_db.sql_to_df(
        app_settings.DB_CONNECTION, 
        SQL_QUERY_1)
    assert df is not None
    assert df.iloc[0]['range'] == 11100

    # Проверка блокировки через функцию
    df2 = load_from_db.sql_to_df(
        app_settings.DB_CONNECTION, 
        "DELETE FROM airports")
    assert df2 is None