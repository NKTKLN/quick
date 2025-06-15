"""Декоратор `duckdb_cache` для кэширования результатов методов.

Кэш сохраняется в DuckDB с использованием Pickle-сериализации.
Ключ формируется на основе аргументов метода и указанных атрибутов экземпляра.
"""

import functools
import logging
from datetime import datetime
from typing import Any, Callable, Dict, TypeVar, cast

from app.db.client import DuckDBClient
from app.settings import ConfigLoader
from app.utils import PickleSerializer

# Настройка логгера
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Callable[..., Any])


def duckdb_cache(*attribute_paths: str) -> Callable[[T], T]:
    """Кэширует результат метода экземпляра класса, используя DuckDB и Pickle.

    При повторном вызове метода с теми же аргументами и значениями указанных атрибутов
    self, результат возвращается из DuckDB, если он уже был сохранён ранее.

    Args:
        *attribute_paths (str): Пути к атрибутам объекта (например, "config.name"),
                                включаемые в кэш-ключ.

    Returns:
        Callable[[T], T]: Декоратор, оборачивающий метод и реализующий кэширование.
    """

    def decorator(method: T) -> T:
        @functools.wraps(method)
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            def extract_nested_attribute(obj: Any, path: str) -> Any:
                """Извлекает значение вложенного атрибута по точечной нотации.

                Args:
                    obj (Any): Объект, у которого нужно извлечь значение.
                    path (str): Путь к атрибуту в формате 'config.name.value'.

                Returns:
                    Any: Значение атрибута.

                Raises:
                    AttributeError: Если путь к атрибуту недоступен.
                """
                try:
                    for attr in path.split("."):
                        obj = getattr(obj, attr)
                    return obj
                except AttributeError as e:
                    raise AttributeError(f"Не удалось получить '{path}': {e}") from e

            config = ConfigLoader.get_config()
            if config.disable_cache:
                result = method(self, *args, **kwargs)
                return result

            serializer = PickleSerializer()
            db_client = DuckDBClient.get_instance()

            # Формирование части ключа из self
            try:
                self_cache_info: Dict[str, Any] = {
                    path: extract_nested_attribute(self, path)
                    for path in attribute_paths
                }
            except AttributeError as e:
                logger.error(f"Не удалось извлечь атрибуты self для ключа: {e}")
                raise

            cache_key_data = {"self": self_cache_info, "args": args, "kwargs": kwargs}

            logger.debug(
                f"Формирование кэш-ключа для метода {method.__name__}: {cache_key_data}"
            )

            key_blob = serializer.dump_key_to_pickle(cache_key_data)

            # Попытка загрузить результат из кэша
            data = db_client.get_by_key(method.__name__, key_blob)
            if data is not None:
                logger.info(
                    "Кэш найден для %s, возвращаем сохранённый результат.",
                    method.__name__,
                )
                return serializer.load_result_from_pickle(
                    data, getattr(self, "_precision", None)
                )

            logger.debug(f"Кэш не найден для {method.__name__}, выполняем метод.")

            result = method(self, *args, **kwargs)

            logger.debug(f"Сохраняем результат метода {method.__name__} в кэш.")
            result_blob = serializer.dump_to_pickle(result)
            db_client.insert_result(
                datetime.now(), method.__name__, key_blob, result_blob
            )

            return result

        return cast(T, wrapper)

    return decorator
