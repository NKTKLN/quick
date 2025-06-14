import logging
import os
import pickle
import threading
from typing import Any, Optional

import duckdb

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
        self, func_name: Any, args_blob: Any, kwargs_blob: Any
    ) -> Any:
        row = self.connection.execute(
            """
        select result from function_calls where function_name = ? and args = ? and kwargs = ?
        """,
            (func_name, args_blob, kwargs_blob),
        ).fetchone()

        if row is None:
            return None

        result_blob = row[0]
        return pickle.loads(result_blob)

    def insert_call(
        self,
        timestamp: Any,
        func_name: Any,
        args_blob: Any,
        kwargs_blob: Any,
        result_blob: Any,
    ) -> None:
        self.connection.execute(
            """
        insert into function_calls (timestamp, function_name, args, kwargs, result)
        values (?, ?, ?, ?, ?)
        """,
            (timestamp, func_name, args_blob, kwargs_blob, result_blob),
        )

    def query(self, sql: str) -> Any:
        return self.connection.execute(sql).fetchdf()

    def close(self) -> None:
        self.connection.close()
