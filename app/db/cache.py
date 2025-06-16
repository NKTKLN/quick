"""Декоратор `duckdb_cache` для кэширования результатов методов.

Кэш сохраняется в DuckDB с использованием Pickle-сериализации.
Ключ формируется на основе аргументов метода и указанных атрибутов экземпляра.
"""

import functools
import logging
from datetime import datetime
from typing import Any, Callable, Dict, TypeVar, cast

from app.common import PickleSerializer
from app.db.client import DuckDBClient
from app.domain import ComputationConfig
from app.settings import ConfigLoader

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
        """Обёртка, которая реализует кэширование результата метода.

        Args:
            method (Callable): Метод экземпляра класса для кэширования.

        Returns:
            Callable: Метод, обёрнутый логикой кэширования.
        """

        @functools.wraps(method)
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            """Выполняет кэширование вызова метода.

            Формирует ключ на основе аргументов метода и указанных атрибутов self,
            пытается загрузить результат из DuckDB. Если результат отсутствует,
            вызывает исходный метод, сохраняет результат и возвращает его.

            Кэширование может быть отключено через конфигурацию.

            Args:
                self: Экземпляр класса, метод которого вызывается.
                *args: Позиционные аргументы метода.
                **kwargs: Именованные аргументы метода.

            Returns:
                Любое: Результат выполнения метода, либо загруженный из кэша.
            """
            config = ConfigLoader.get_config()
            computation_config: ComputationConfig = self.config
            if computation_config.disable_cache or config.disable_cache:
                result = method(self, *args, **kwargs)
                return result

            serializer = PickleSerializer()
            db_client = DuckDBClient.get_instance()

            # Формирование части ключа из self
            try:
                self_cache_info: Dict[str, Any] = {
                    path: _extract_nested_attribute(self, path)
                    for path in attribute_paths
                }
            except AttributeError as e:
                logger.error(f"Не удалось извлечь атрибуты self для ключа: {e}")
                raise

            cache_key_data = {"self": self_cache_info, "args": args, "kwargs": kwargs}

            logger.debug(
                f"Формирование кэш-ключа для метода {method.__name__}: {cache_key_data}"
            )

            try:
                key_blob = serializer.dump_key_to_pickle(cache_key_data)
            except Exception as e:
                logger.warning(f"Ошибка сериализации ключа для {method.__name__}: {e}")
                return method(self, *args, **kwargs)

            key_hash = key_blob[:8].hex()

            # Попытка загрузить результат из кэша
            try:
                data = db_client.get_by_key(method.__name__, key_blob)
                if data is not None:
                    logger.info(
                        f"Кэш найден для {method.__name__} (key={key_hash}), "
                        "возвращаем результат."
                    )
                    return serializer.load_result_from_pickle(
                        data, getattr(self, "_precision", None)
                    )
            except Exception as e:
                logger.warning(
                    f"Ошибка при загрузке кэша для {method.__name__} "
                    f"(key={key_hash}): {e}"
                )

            logger.debug(f"Кэш не найден для {method.__name__}, выполняем метод.")
            result = method(self, *args, **kwargs)

            try:
                logger.debug(f"Сохраняем результат метода {method.__name__} в кэш.")
                result_blob = serializer.dump_to_pickle(result)
                db_client.insert_result(
                    datetime.now(), method.__name__, key_blob, result_blob
                )
            except Exception as e:
                logger.warning(
                    f"Ошибка при сохранении результата в кэш для {method.__name__} "
                    f"(key={key_hash}): {e}"
                )

            return result

        return cast(T, wrapper)

    return decorator


def _extract_nested_attribute(obj: Any, path: str) -> Any:
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
