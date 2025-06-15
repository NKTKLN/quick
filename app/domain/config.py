"""Модуль параметров инициализации конфигурации.

Содержит класс ConfigInitParams — структуру данных с параметрами,
которые можно передать для инициализации конфигурации приложения.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ConfigInitParams:
    """Параметры инициализации конфигурации приложения.

    Все поля опциональны, используются для установки соответствующих
    переменных окружения в ConfigLoader.

    Атрибуты:
        disable_cache (Optional[bool]): Отключение кэширования.
        duckdb_path (Optional[str]): Путь к DuckDB базе данных.
        disable_logging (Optional[bool]): Отключение логирования.
        log_level (Optional[str]): Уровень логирования.
        log_path (Optional[str]): Путь для логов.
        port (Optional[int]): Порт приложения.
    """

    disable_cache: Optional[bool] = None
    duckdb_path: Optional[str] = None
    disable_logging: Optional[bool] = None
    log_level: Optional[int] = None
    log_path: Optional[str] = None
    port: Optional[int] = None
