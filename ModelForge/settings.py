from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

env_file: str = ".env"
load_dotenv(env_file)

class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_file=env_file, env_file_encoding="utf-8")

    DEBUG: bool = True
    # Путь, где необходимо разместить наборы данных для 
    # Запуска тестов
    TEST_DATASETS_PATH: str = "./datasets/"
    # Строка с данными для подключения к data base
    DB_CONNECTION: str
    AIO_DB_CONNECTION: str
    KAGGLE_USERNAME: str
    KAGGLE_KEY: str


settings = Settings() #type:ignore


