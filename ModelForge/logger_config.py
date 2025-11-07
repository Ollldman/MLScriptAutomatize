import logging
import os
from datetime import datetime

def setup_logging():
    """
    Модуль инициализации логгера для всего приложения.
    Обязательно соблюдать порядок импорта и инициализации.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"app_{timestamp}.log")

    # Создаём форматтер
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)


    if root_logger.hasHandlers():
        root_logger.handlers.clear()


    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.info(f"Логирование настроено. Лог-файл: {log_file}")