"""Модуль параметров инициализации конфигурации.

Содержит класс ConfigInitParams — структуру данных с параметрами, которые можно передать
при запуске для настройки конфигурации приложения.
"""

from dataclasses import dataclass


@dataclass
class ConfigInitParams:
    """Параметры инициализации конфигурации приложения.

    Используются для предварительной установки переменных окружения
    и параметров поведения при инициализации конфигурации в ConfigLoader.

    Attributes:
        disable_cache (bool | None): Признак отключения кэширования.
        duckdb_path (str | None): Путь до файла базы данных DuckDB.
        disable_logging (bool | None): Признак отключения логирования.
        log_level (str | None): Строковый уровень логирования.
        log_path (str | None): Путь к файлу логов, если логирование включено.
    """

    disable_cache: bool | None = None
    duckdb_path: str | None = None
    disable_logging: bool | None = None
    log_level: str | None = None
    log_path: str | None = None
