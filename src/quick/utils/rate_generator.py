"""Модуль генерации матриц вероятностей p^(0) и p^(1) для MAP-потоков.

Предоставляет функцию для создания случайных матриц вероятностей переходов
без генерации заявки p^(0) и с генерацией заявки p^(1), используемых в MAP-потоке.
"""

import numpy as np
from loguru import logger
from numpy.typing import NDArray


def map_intensity_matrix_generator(
    n: int,
    p_max: float | None = None,
    q_max: float | None = None,
    depth: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Генерирует случайные матрицы вероятностей p^(0) и p^(1).

    Args:
        n (int): Размерность квадратных матриц.
        p_max (float | None): Максимальное значение элементов p. По умолчанию 0.5/(n-1).
        q_max (float | None): Максимальное значение элементов q. По умолчанию 0.5/(n-1).
        depth (int): Глубина рекурсии при повторном вызове. По умолчанию 10.

    Raises:
        ValueError: Если depth <= 0.

    Returns:
        tuple[NDArray, NDArray]: Кортеж из матриц p и q.
    """
    if depth <= 0:
        logger.error("Достигнута максимальная глубина рекурсии (depth).")
        raise ValueError("Достигнута максимальная глубина рекурсии (depth).")

    if n <= 0:
        raise ValueError("Размерность MAP-процесса должна быть положительной.")

    # Для единственной MAP-фазы переход без генерации заявки невозможен,
    # а единственный допустимый переход с генерацией заявки имеет
    # вероятность 1. Это точно удовлетворяет условию нормировки (1).
    if n == 1:
        return (
            np.zeros((1, 1), dtype=np.float64),
            np.ones((1, 1), dtype=np.float64),
        )

    real_p_max = 0.5 / (n - 1)

    if p_max is None:
        p_max = real_p_max
    if q_max is None:
        q_max = real_p_max

    logger.info(
        "Генерируем матрицы вероятностей MAP-потока с параметрами: "
        f"{n=}, {p_max=}, {q_max=}, {depth=}"
    )

    p = np.random.uniform(0, p_max, size=(n, n))
    np.fill_diagonal(p, 0)

    q = np.random.uniform(0, q_max, size=(n, n))
    q[:, -1] = 1 - p.sum(axis=1) - q[:, :-1].sum(axis=1)

    logger.debug(f"Сгенерирована матрица p:\n{np.round(p, 4)}")
    logger.debug(f"Сгенерирована матрица q:\n{np.round(q, 4)}")

    if np.any(q[:, -1] < 0):
        logger.warning(
            "Матрица q некорректна (отрицательные элементы в последнем столбце), "
            "пересчитываем..."
        )
        return map_intensity_matrix_generator(n, p_max, q_max, depth - 1)

    logger.info("Матрицы вероятностей MAP-переходов успешно сгенерированы")
    return p, q
