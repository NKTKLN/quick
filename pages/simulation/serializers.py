"""Содержит функции для сериализации результатов моделирования в JSON.

Модуль отвечает за:
    - преобразование numpy-массивов и numpy-скаляров в стандартные Python-типы;
    - рекурсивную обработку вложенных структур данных;
    - замену невалидных JSON-значений, таких как NaN и Inf, на None;
    - формирование JSON-представления результатов моделирования.

Основные функции:
    numpy_to_python() — рекурсивно преобразует numpy-объекты в типы,
    совместимые с JSON.

    results_to_json() — подготавливает результаты моделирования и временной
    массив к сериализации и возвращает JSON-строку.
"""

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray


def numpy_to_python(obj: Any) -> Any:
    """Рекурсивно преобразует numpy-объекты в стандартные Python-типы.

    Преобразует:
    - numpy.ndarray → list
    - numpy числа → Python числа
    - numpy bool → bool
    - NaN и Inf → None

    Args:
        obj (Any): Объект, который может содержать numpy-типы.

    Returns:
        Any: Объект, содержащий только стандартные Python-типы.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer | np.floating | np.bool_):
        return obj.item()
    if isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    if isinstance(obj, list | tuple):
        return [numpy_to_python(item) for item in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def results_to_json(results: dict, time_array: NDArray) -> str:
    """Преобразует результаты моделирования в JSON-строку.

    Args:
        results (dict): Результаты моделирования.
        time_array (np.ndarray): Массив временных точек.

    Returns:
        str: JSON-представление результатов.
    """
    serializable = numpy_to_python(results)
    serializable["time_array"] = numpy_to_python(time_array)
    return json.dumps(
        serializable,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
