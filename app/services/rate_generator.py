"""Модуль генерации интенсивностных матриц p и q для MAP-потоков.

Предоставляет функцию для создания случайных матриц p и q, используемых в
марковских пуассоновских потоках (MAP) при моделировании систем массового
обслуживания. Матрицы нормируются по строкам и соответствуют требованиям
вероятностной интерпретации.
"""

from math import floor
from typing import Optional

import numpy as np


def map_intensity_matrix_generator(
    n: int,
    p_max: Optional[float] = None,
    q_max: Optional[float] = None,
    eps: float = 1e-2,
    depth: int = 10,
) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    """Генерирует случайные матрицы p и q с нормировкой строк.

    Args:
        n (int): Размерность квадратных матриц.
        p_max (Optional[float]): Максимальное значение элементов p. По умолчанию None.
        q_max (Optional[float]): Максимальное значение элементов q. По умолчанию None.
        eps (float): Отступ
        depth (int): Глубина рекурсии при повторном вызове. По умолчанию 5.

    Raises:
        ValueError: Если depth <= 0.

    Returns:
        tuple[np.ndarray, np.ndarray]: Кортеж из матриц p и q.
    """
    if depth <= 0:
        raise ValueError("Достигнута максимальная глубина рекурсии (depth).")

    factor = 1e2
    real_p_max = floor((1 / (n - 1)) * factor) / factor - eps
    print(real_p_max)

    if p_max is None:
        p_max = real_p_max
    if q_max is None:
        q_max = real_p_max

    p = np.random.uniform(0, p_max, size=(n, n))
    np.fill_diagonal(p, 0)

    q = np.random.uniform(0, q_max, size=(n, n))
    q[:, -1] = 1 - p.sum(axis=1) - q[:, :-1].sum(axis=1)

    print(q, q_max, depth)

    if np.any(q[:, -1] < 0):
        return map_intensity_matrix_generator(n, p_max, q_max, eps, depth - 1)

    return p, q
