"""Пакет для работы с кэшированием в DuckDB."""

from .cache import duckdb_cache
from .client import DuckDBClient

__all__ = ["duckdb_cache", "DuckDBClient"]
