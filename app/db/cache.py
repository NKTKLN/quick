import logging
from datetime import datetime
from pickle import dumps
from app.utils import PickleSerializer
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from app.db.client import DuckDBClient
from app.domain import CachingType

# Настройка логгера
logger = logging.getLogger(__name__)

def duckdb_cache(caching_type: CachingType, *attrs: Any):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(self, *args: Any, **kwargs: Any) -> NDArray[np.float64] | Any:
            logger.debug(f"Вызов кеширующего декоратора для функции '{func.__name__}' с args={args} kwargs={kwargs}")
            
            def get_nested_attr(obj, attr_path):
                attrs = attr_path.split('.')
                for attr in attrs:
                    obj = getattr(obj, attr)
                return obj

            db = DuckDBClient.get_instance()
            s = PickleSerializer()

            key_parts = []
            for attr_path in attrs:
                value = get_nested_attr(self, attr_path)
                key_parts.append(value)
            self_blob = dumps(tuple(key_parts))
            args_blob = s.dump_args_to_pickle(args)
            kwargs_blob = s.dump_kwargs_to_pickle(kwargs)
            logger.debug("Сериализованы self, args и kwargs")

            data = db.get_existing_call(func.__name__, self_blob, args_blob, kwargs_blob, s.code_word, caching_type)
            if data is not None:
                logger.info(f"Кеш найден для функции '{func.__name__}'")
                return s.load_result_from_pickle(data, getattr(self, "_precision", None))
            logger.info(f"Кеша нет для функции '{func.__name__}', выполняем функцию")
            
            result = func(self, *args, **kwargs)
            logger.debug(f"Функция '{func.__name__}' выполнена, кешируем результат")

            result_blob = s.dump_result_to_pickle(result)
            db.insert_call(
                datetime.now(), func.__name__, self_blob, args_blob, kwargs_blob, result_blob
            )
            logger.debug(f"Результат сохранён в кеш в БД для функции '{func.__name__}'")

            return result
        return wrapper
    return decorator
