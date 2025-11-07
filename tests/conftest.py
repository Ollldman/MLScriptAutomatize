import sys
import os
import pytest

# adding project root to PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ModelForge.settings import settings, Settings

@pytest.fixture(scope='session')
def app_settings() -> Settings:
    """Фикстура для доступа к настройкам приложения в тестах."""
    return settings