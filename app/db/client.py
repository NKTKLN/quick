import logging
import os
import threading
from typing import Any, Optional

import duckdb

from app.domain.config import CachingType
from app.settings import ConfigLoader

# Setting up logging
logger = logging.getLogger(__name__)


class DuckDBClient:
    _instance: Optional["DuckDBClient"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        config = ConfigLoader.get_config()
        self.connection = duckdb.connect(database=config.duckdb_path, read_only=False)
        self.connection.execute("PRAGMA threads=4")
        self._init_schema()

    def _init_schema(self) -> None:  # TODO: add raise
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        with open(f"{current_dir}/schema.sql", "r") as f:
            schema_sql = f.read()
            self.connection.execute(schema_sql)

    @classmethod
    def get_instance(cls) -> "DuckDBClient":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get_existing_call(
        self, func_name: Any, self_blob: Any, args_blob: Any, kwargs_blob: Any, code_word: Any, caching_type: CachingType
    ) -> Any:
        match caching_type:
            case CachingType.SELF:
                row = self.connection.execute(
                    """
                select result from function_calls where function_name = ? and self_data = ? and args = ? and kwargs = ?
                """,
                    (func_name, self_blob, code_word, code_word),
                ).fetchone()
            case CachingType.ARGS:
                row = self.connection.execute(
                    """
                select result from function_calls where function_name = ? and args = ? and kwargs = ?
                """,
                    (func_name, args_blob, kwargs_blob),
                ).fetchone()
            case CachingType.MERGED:
                row = self.connection.execute(
                    """
                select result from function_calls where function_name = ? and self_data = ? and args = ? and kwargs = ?
                """,
                    (func_name, self_blob, args_blob, kwargs_blob),
                ).fetchone()
            case _:
                raise ValueError

        if row is None:
            return None

        return row[0]

    def insert_call(
        self,
        timestamp: Any,
        func_name: Any,
        self_blob: Any,
        args_blob: Any,
        kwargs_blob: Any,
        result_blob: Any,
    ) -> None:
        self.connection.execute(
            """
        insert into function_calls (timestamp, function_name, self_data, args, kwargs, result)
        values (?, ?, ?, ?, ?, ?)
        """,
            (timestamp, func_name, self_blob, args_blob, kwargs_blob, result_blob),
        )

    def query(self, sql: str) -> Any:
        return self.connection.execute(sql).fetchdf()

    def close(self) -> None:
        self.connection.close()
