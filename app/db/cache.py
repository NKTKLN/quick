import logging
from datetime import datetime
from pickle import dumps
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from app.db.client import DuckDBClient

# Setting up logging
logger = logging.getLogger(__name__)


def duckdb_cache(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> NDArray[np.float64] | Any:
        db = DuckDBClient.get_instance()
        args_blob = dumps(args)
        kwargs_blob = dumps(kwargs)

        data = db.get_existing_call(func.__name__, args_blob, kwargs_blob)
        if data is not None:
            return data

        result = func(*args, **kwargs)
        result_blob = dumps(result)
        db.insert_call(
            datetime.now(), func.__name__, args_blob, kwargs_blob, result_blob
        )
        return result

    return wrapper
