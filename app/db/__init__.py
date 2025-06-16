"""Пакет для работы с кэшированием в DuckDB.

Импортирует основные компоненты:
- duckdb_cache — декоратор для кэширования методов.
- DuckDBClient — клиент для взаимодействия с DuckDB.
"""

from .cache import duckdb_cache
from .client import DuckDBClient

__all__ = ["duckdb_cache", "DuckDBClient"]
