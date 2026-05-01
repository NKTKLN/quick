"""Модуль генераторов матриц D по MAP-параметрам..

Реализует функции map_d_0_matrix_generator и map_d_1_matrix_generator, которые
генерируют матрицы D₀ и D₁ для параметров потока.
"""

import numpy as np
from loguru import logger
from numpy.typing import NDArray


def map_d_0_matrix_generator(
    p_rate: NDArray[np.float64], lambda_rate: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Генерирует матрицу D₀ по MAP-параметрам.

    Args:
        p_rate (NDArray[np.float64]): Матрица интенсивностей обслуживания.
        lambda_rate (NDArray[np.float64]): Интенсивность поступления заявок (λ > 0).

    Returns:
        NDArray[np.float64]: Матрица D₀ для текущих параметров потока.
    """
    logger.debug("Генерация матрицы D₀ из p_rate и λ.")
    matrix = p_rate.copy()
    matrix *= lambda_rate[:, np.newaxis]
    np.fill_diagonal(matrix, -lambda_rate)
    logger.debug(f"Матрица D₀ сгенерирована. {matrix.shape=}")
    return matrix


def map_d_1_matrix_generator(
    q_rate: NDArray[np.float64], lambda_rate: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Генерирует матрицу D₁ по MAP-параметрам.

    Args:
        q_rate (NDArray[np.float64]): Матрица интенсивностей поступления.
        lambda_rate (NDArray[np.float64]): Интенсивность поступления заявок (λ > 0).

    Returns:
        NDArray[np.float64]: Матрица D₁ для текущих параметров потока.
    """
    logger.debug("Генерация матрицы D₁ из q_rate и λ.")
    matrix = q_rate.copy()
    matrix *= lambda_rate[:, np.newaxis]
    logger.debug(f"Матрица D₁ сгенерирована. {matrix.shape=}")
    return matrix
