"""Модуль параметров инициализации конфигурации.

Содержит класс ConfigInitParams — структуру данных с параметрами,
которые можно передать при запуске для настройки конфигурации приложения.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ConfigInitParams:
    """Параметры инициализации конфигурации приложения.

    Используются для предварительной установки переменных окружения
    и параметров поведения при инициализации конфигурации в ConfigLoader.

    Attributes:
        disable_cache (Optional[bool]): Признак отключения кэширования.
        duckdb_path (Optional[str]): Путь до файла базы данных DuckDB.
        disable_logging (Optional[bool]): Признак отключения логирования.
        log_level (Optional[int]): Целочисленный уровень логирования.
        log_path (Optional[str]): Путь к файлу логов, если логирование включено.
    """

    disable_cache: Optional[bool] = None
    duckdb_path: Optional[str] = None
    disable_logging: Optional[bool] = None
    log_level: Optional[int] = None
    log_path: Optional[str] = None
