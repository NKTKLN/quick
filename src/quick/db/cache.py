"""Декоратор `duckdb_cache` для кэширования результатов методов.

Кэш сохраняется в DuckDB с использованием Pickle-сериализации. Ключ формируется на
основе аргументов метода и указанных атрибутов экземпляра.
"""

import functools
from collections.abc import Callable
from datetime import datetime
from pickle import PickleError, UnpicklingError
from typing import Any, TypeVar, cast

from duckdb import Error as DuckDBError
from loguru import logger

from quick.db.client import DuckDBClient
from quick.domain import ComputationConfig

# from quick.services.solvers.base import BasicProbabilitySolver
from quick.services.solvers.base import BasicProbabilitySolver
from quick.settings import ConfigLoader
from quick.utils import PickleSerializer

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

            Raises:
                TypeError: Если декорируемый метод вызван не у экземпляра
                    BasicProbabilitySolver.
                AttributeError: Если один или несколько путей из attribute_paths
                    не могут быть извлечены из объекта self.
            """
            config = ConfigLoader.get_config()

            if not isinstance(self, BasicProbabilitySolver):
                logger.error(
                    "Некорректный тип self: "
                    f"ожидался BasicProbabilitySolver, получен {type(self)}"
                )
                raise TypeError(
                    "Декорируемый метод должен принадлежать классу "
                    "BasicProbabilitySolver"
                )

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

            key_blob = _build_key_blob(
                self,
                args,
                kwargs,
                attribute_paths,
                method.__name__,
            )

            if key_blob is None:
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


def _build_key_blob(
    self: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    attribute_paths: tuple[str, ...],
    method_name: str,
) -> bytes | None:
    """Формирует сериализованный кэш-ключ для метода.

    Args:
        self (Any): Экземпляр класса, для которого вызывается метод.
        args (tuple[Any, ...]): Позиционные аргументы метода.
        kwargs (dict[str, Any]): Именованные аргументы метода.
        attribute_paths (tuple[str, ...]): Пути к атрибутам `self`,
            включаемые в кэш-ключ (например, "config.name").
        method_name (str): Имя метода (используется для логирования).

    Returns:
        bytes | None: Сериализованный ключ (pickle blob) или `None`,
            если произошла ошибка сериализации.

    Raises:
        AttributeError: Если один из путей `attribute_paths`
            не может быть извлечён из `self`.
    """
    try:
        self_cache_info = {
            path: _extract_nested_attribute(self, path) for path in attribute_paths
        }
    except AttributeError as e:
        logger.error(f"Не удалось извлечь атрибуты self для ключа: {e}")
        raise AttributeError(
            "Не удалось извлечь один или несколько атрибутов из self "
            f"для формирования кэш-ключа: {attribute_paths}"
        ) from e

    cache_key_data = {"self": self_cache_info, "args": args, "kwargs": kwargs}

    logger.debug(f"Формирование кэш-ключа для метода {method_name}: {cache_key_data}")

    try:
        return PickleSerializer.dump_key_to_pickle(cache_key_data)
    except (PickleError, TypeError) as e:
        logger.warning(f"Ошибка сериализации ключа для {method_name}: {e}")
        return None


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
