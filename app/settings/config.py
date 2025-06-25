"""Модуль конфигурации приложения.

Определяет классы и методы для загрузки, инициализации и кеширования
конфигурационных параметров приложения с использованием библиотеки pydantic.

Основные компоненты:
    - AppConfig: Класс конфигурации, загружаемый из переменных окружения или .env файла.
    - ConfigLoader: Потокобезопасный загрузчик конфигурации, позволяющий
      инициализировать параметры из объекта и кешировать результаты.
"""

import os
import threading
from dataclasses import asdict

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.domain import ConfigInitParams


class AppConfig(BaseSettings):
    """Конфигурация приложения, загружаемая из переменных окружения или файла .env.

    Атрибуты:
        disable_cache (bool): Отключение кэширования (по умолчанию False).
        duckdb_path (str): Путь к базе данных DuckDB (по умолчанию "cache_data.duckdb").
        disable_logging (bool): Отключение логирования (по умолчанию False).
        log_level (str): Уровень логирования (по умолчанию "INFO").
        log_path (str): Путь для записи логов (по умолчанию пустая строка).
        log_format (str): Формат лог-сообщений.
    """

    disable_cache: bool = Field(default=False)
    duckdb_path: str = Field(default="cache_data.duckdb")
    disable_logging: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    log_path: str = Field(default="")

    log_format: str = (
        "<cyan>[{time:DD/MM/YY HH:mm:ss}]</cyan> "
        "<light-magenta>[{file}:{function}:{line}]</light-magenta> "
        "<lvl>[{level}]</lvl> - {message}"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",  # Без префикса для переменных окружения
    )
    _ = model_config


class ConfigLoader:
    """Потокобезопасный загрузчик и кешировщик конфигурации приложения."""

    _instance: AppConfig | None = None
    _lock = threading.Lock()

    @classmethod
    def init(cls, params: ConfigInitParams) -> None:
        """Инициализирует конфигурацию приложения.

        Устанавливает переменные окружения из переданных параметров,
        очищает кеш get_config для повторной загрузки с новыми значениями.

        Args:
            params (ConfigInitParams): Объект с параметрами конфигурации.
        """
        with cls._lock:
            for key, value in asdict(params).items():
                env_key = key.upper()
                os.environ[env_key] = str(value)
            cls._instance = None

    @classmethod
    def get_config(cls) -> AppConfig:
        """Возвращает инстанцию AppConfig с текущими настройками.

        Использует кеширование для избежания повторных загрузок конфигурации.

        Returns:
            AppConfig: Объект с конфигурацией приложения.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = AppConfig()
            return cls._instance
