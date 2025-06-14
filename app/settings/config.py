import os
import threading
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    duckdb_path: str = Field(default="cache_data.duckdb")
    enable_cache: bool = Field(default=True)
    log_level: str = Field(default="info")
    port: int = Field(default=8080)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
    )


class ConfigLoader:
    _config: Optional[AppConfig] = None
    _lock = threading.Lock()

    @classmethod
    def init(
        cls,
        duckdb_path: Optional[str] = None,
        enable_cache: Optional[bool] = None,
        log_level: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        if duckdb_path is not None:
            os.environ["DUCKDB_PATH"] = duckdb_path
        if enable_cache is not None:
            os.environ["ENABLE_CACHE"] = str(enable_cache)
        if log_level is not None:
            os.environ["LOG_LEVEL"] = log_level
        if port is not None:
            os.environ["PORT"] = str(port)

        with cls._lock:
            cls._config = None

    @classmethod
    def get_config(cls) -> AppConfig:
        with cls._lock:
            if cls._config is None:
                cls._config = AppConfig()
            return cls._config
