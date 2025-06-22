"""Клиент для работы с DuckDB, реализующий паттерн singleton.

Обеспечивает подключение к базе, инициализацию схемы, кэширование результатов
функций по ключу и управление соединением.
"""

import logging
import os
import threading
from datetime import datetime
from typing import Any, Optional

import duckdb

from app.settings import ConfigLoader

# Настройка логгера
logger = logging.getLogger(__name__)


class DuckDBClient:
    """Singleton-класс для взаимодействия с DuckDB.

    Обеспечивает подключение к базе, инициализацию схемы, операции чтения и записи
    результатов функций по ключу.

    Attributes:
        _instance (Optional[DuckDBClient]): Singleton-экземпляр класса.
        _lock (threading.Lock): Блокировка для потокобезопасного создания экземпляра.
    """

    _instance: Optional["DuckDBClient"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """Инициализирует подключение к DuckDB и схему базы."""
        if hasattr(self, "_initialized") and self._initialized:
            return
        config = ConfigLoader.get_config()
        self.connection = duckdb.connect(database=config.duckdb_path, read_only=False)
        self.connection.execute("PRAGMA threads=4")
        logger.info("Установлено соединение с DuckDB, инициализация схемы.")
        self._init_schema()
        self._initialized: bool = True

    def _init_schema(self) -> None:
        """Инициализирует схему базы данных, выполняя SQL из файла schema.sql.

        Raises:
            FileNotFoundError: Если файл schema.sql не найден.
            RuntimeError: Если произошла ошибка при выполнении SQL схемы.
        """
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        schema_path = os.path.abspath(
            os.path.join(current_dir, "..", "sql", "schema.sql")
        )

        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Файл схемы не найден: {schema_path}")

        try:
            with open(schema_path, "r") as f:
                schema_sql = f.read()
                self.connection.execute(schema_sql)
                logger.info("Схема базы данных успешно инициализирована.")
        except Exception as exc:
            raise RuntimeError(
                f"Ошибка при инициализации схемы из {schema_path}"
            ) from exc

    def __new__(cls, *args: Any, **kwargs: Any) -> "DuckDBClient":
        """Возвращает singleton-экземпляр DuckDBClient.

        Создаёт новый экземпляр, если он ещё не был создан.

        Returns:
            DuckDBClient: Singleton-экземпляр клиента.
        """
        with cls._lock:
            if cls._instance is None:
                logger.debug("Создание нового singleton-экземпляра DuckDBClient.")
                cls._instance = super().__new__(cls)
            else:
                logger.debug(
                    "Используется существующий singleton-экземпляр DuckDBClient."
                )
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Сбрасывает singleton-экземпляр и закрывает соединение, если существует."""
        with cls._lock:
            if cls._instance is not None:
                logger.debug("Сброс singleton-экземпляра DuckDBClient.")
                cls._instance.close()
                cls._instance = None

    def get_by_key(self, func_name: str, key_blob: bytes) -> Optional[bytes]:
        """Получает результат из базы по имени функции и бинарному ключу.

        Args:
            func_name (str): Имя функции/метода, для которого ищется кэш.
            key_blob (bytes): Сериализованный ключ вызова.

        Returns:
            Optional[bytes]: Данные результата из кэша или None, если не найдено.
        """
        logger.debug(f"Выполняется поиск кэша для функции '{func_name}'.")
        result = self.connection.execute(
            """select result from function_calls \
            where function_name = ? and key_data = ?
            """,
            (func_name, key_blob),
        ).fetchone()
        return result[0] if result else None

    def insert_result(
        self, timestamp: datetime, func_name: str, key_blob: bytes, result_blob: bytes
    ) -> None:
        """Сохраняет результат выполнения функции в базу.

        Args:
            timestamp (datetime): Время сохранения результата.
            func_name (str): Имя функции/метода.
            key_blob (bytes): Сериализованный ключ вызова.
            result_blob (bytes): Сериализованные данные результата.
        """
        logger.debug(f"Сохраняется результат для функции '{func_name}' на {timestamp}.")
        self.connection.execute(
            """insert into function_calls (timestamp, function_name, key_data, result) \
            values (?, ?, ?, ?)
            """,
            (timestamp, func_name, key_blob, result_blob),
        )
        self.connection.commit()

    def close(self) -> None:
        """Закрывает соединение с базой DuckDB."""
        logger.debug("Закрывается соединение с DuckDB.")
        self.connection.close()

    def __del__(self) -> None:
        """Автоматически закрывает соединение при уничтожении объекта."""
        try:
            self.close()
        except Exception as e:
            logging.error(f"Ошибка закрытия соединения: {e}")
