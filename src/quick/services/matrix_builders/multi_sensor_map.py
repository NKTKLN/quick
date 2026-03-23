"""Модуль построения матрицы коэффициентов для СМО с Марковскими входными потоками и несколькими датчиками.

Содержит класс MultiSensorMAPServerMatrixBuilder, который строит матрицу коэффициентов
систем массового обслуживания с несколькими датчикаминесколькими датчиками, в которых
входной поток описывается марковским процессом (MAP). Основан на параметрах переходов
между состояниями MAP и интенсивностях обслуживания и ухода.
"""

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.domain.models import MAPSystemParams
from quick.services.matrix_builders.base import BaseMAPMatrixBuilder


class MultiSensorMAPServerMatrixBuilder(BaseMAPMatrixBuilder):
    """Построитель матрицы коэффициентов для СМО с Марковскими входными потоками и несколькими датчиками.

    Использует матрицы вероятностей переходов p_rate и q_rate для построения
    четырёхмерной матрицы, сворачиваемой в двухмерную.
    """

    def build(self) -> NDArray[np.float64]:
        """Формирует матрицу коэффициентов для СМО с Марковскими входными потоками.

        Returns:
            NDArray[np.float64]: Квадратная матрица коэффициентов (n² x n²).

        Raises:
            ValueError: Если базовые параметры системы не являются MAPSystemParams.
        """
        if not isinstance(self.params, MAPSystemParams):
            logger.error("Базовые параметры не являются экземпляром MAPSystemParams")
            raise ValueError(
                "Базовые параметры должны быть экземпляром MAPSystemParams"
            )

        logger.info("Начато построение матрицы коэффициентов.")
        n = self.params.max_customers
        μ, ν = self.params.mu_rate, self.params.nu_rate

        d_0_t = self._d_0_matrix_generator().T
        d_1_t = self._d_1_matrix_generator().T

        if d_0_t.shape != d_1_t.shape or d_0_t.shape[0] != d_0_t.shape[1]:
            logger.error(
                "Матрицы D₀ и D₁ должны быть одинакового размера и быть квадратными."
            )
            raise ValueError(
                "Матрицы D₀ и D₁ должны быть одинакового размера и быть квадратными."
            )

        m = d_0_t.shape[0]

        coefficients_matrix = np.zeros((n, n, m, m), dtype=np.float64)

        for index in range(n):
            if index == 0:
                coefficients_matrix[index, index] = d_0_t
                coefficients_matrix[index, index + 1] = μ * np.eye(m)
            elif index == n - 1:
                coefficients_matrix[index, index - 1] = d_1_t
                coefficients_matrix[index, index] = (
                    d_0_t + d_1_t - (μ + (index - 1) * ν) * np.eye(m)
                )
            else:
                coefficients_matrix[index, index - 1] = d_1_t
                coefficients_matrix[index, index] = d_0_t - (
                    μ + ν * max(0, index - 1)
                ) * np.eye(m)
                coefficients_matrix[index, index + 1] = (μ + index * ν) * np.eye(m)

        logger.debug("Преобразование 4D матрицы в 2D представление.")
        coefficients_matrix = coefficients_matrix.transpose(0, 2, 1, 3).reshape(
            m * n, m * n
        )  # type: ignore

        logger.success(
            "Матрица коэффициентов для СМО с Марковскими входными потоками и несколькими датчикам "
            "успешно сгенерирована."
        )
        return coefficients_matrix
