from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

env_file: str = ".env"
load_dotenv(env_file)

class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_file=env_file, env_file_encoding="utf-8")

    DEBUG: bool = True
    TEST_DATASETS_PATH: str = "./datasets/"


settings = Settings() #type:ignore


