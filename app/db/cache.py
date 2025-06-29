"""Декоратор `duckdb_cache` для кэширования результатов методов.

Кэш сохраняется в DuckDB с использованием Pickle-сериализации.
Ключ формируется на основе аргументов метода и указанных атрибутов экземпляра.
"""

import functools
from datetime import datetime
from pickle import PickleError, UnpicklingError
from typing import Any, Callable, TypeVar, cast

from duckdb import Error as DuckDBError
from loguru import logger

from app.db.client import DuckDBClient
from app.domain import ComputationConfig
from app.settings import ConfigLoader
from app.utils import PickleSerializer

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
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Выполняет кэширование вызова метода.

            Формирует ключ на основе аргументов метода и указанных атрибутов self,
            пытается загрузить результат из DuckDB. Если результат отсутствует,
            вызывает исходный метод, сохраняет результат и возвращает его.

            Кэширование может быть отключено через конфигурацию.

            Args:
                self (Any): Экземпляр класса решателя.
                *args (Any): Позиционные аргументы метода.
                **kwargs (Any): Именованные аргументы метода.

            Returns:
                Any: Результат выполнения метода, либо загруженный из кэша.
            """
            config = ConfigLoader.get_config()
            computation_config: ComputationConfig = self.config
            if computation_config.disable_cache or config.disable_cache:
                logger.debug(
                    f"Кэш отключён для {method.__name__}, выполнение без "
                    "использования кэша."
                )
                result = method(self, *args, **kwargs)
                return result

            db_client = DuckDBClient()

            logger.debug(f"Запрос к кэшу для метода: {method.__name__}")

            # Формирование части ключа из self
            try:
                self_cache_info: dict[str, Any] = {
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
                key_blob = PickleSerializer.dump_key_to_pickle(cache_key_data)
            except (PickleError, TypeError) as e:
                logger.warning(f"Ошибка сериализации ключа для {method.__name__}: {e}")
                return method(self, *args, **kwargs)

            # Попытка загрузить результат из кэша
            try:
                data = db_client.get_by_key(method.__name__, key_blob)
                if data is not None:
                    result = PickleSerializer.load_result_from_pickle(
                        data, getattr(self, "_precision", None)
                    )
                    if result is not None:
                        logger.info(
                            f"Кэш найден для {method.__name__}, возвращаем результат."
                        )
                        return result
                    logger.debug(
                        "Кэш был найден, но десериализация результата вернула None "
                        f"для {method.__name__}"
                    )
            except (DuckDBError, UnpicklingError, TypeError) as e:
                logger.warning(f"Ошибка при загрузке кэша для {method.__name__}: {e}")

            logger.debug(f"Кэш не найден для {method.__name__}, выполняем метод.")
            result = method(self, *args, **kwargs)

            try:
                logger.debug(f"Сохраняем результат метода {method.__name__} в кэш.")
                result_blob = PickleSerializer.dump_to_pickle(result)
                db_client.insert_result(
                    datetime.now(), method.__name__, key_blob, result_blob
                )
            except (DuckDBError, PickleError, TypeError) as e:
                logger.warning(
                    f"Ошибка при сохранении результата в кэш для {method.__name__}: {e}"
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
