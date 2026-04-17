"""Модуль построения матрицы коэффициентов для СМО с Марковскими входными потоками и несколькими датчиками.

Содержит класс MultiSensorMAPServerMatrixBuilder, который строит матрицу коэффициентов
систем массового обслуживания с несколькими датчикаминесколькими датчиками, в которых
входной поток описывается марковским процессом (MAP). Основан на параметрах переходов
между состояниями MAP и интенсивностях обслуживания и ухода.
"""

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.domain.models.system_params import MAPSystemParams
from quick.services.matrix_builders.base import BaseMatrixBuilder


class MultiSensorMAPServerMatrixBuilder(BaseMatrixBuilder):
    """Построитель матрицы коэффициентов для СМО с Марковскими входными потоками и несколькими датчиками.

    Использует матрицы вероятностей переходов p_rate и q_rate для построения
    четырёхмерной матрицы, сворачиваемой в двухмерную.
    """

    def __init__(self, params: MAPSystemParams) -> None:
        """Инициализирует базовый построитель с параметрами модели.

        Args:
            params (MAPSystemParams): Параметры модели СМО.
        """
        self.params = params

        logger.debug(
            f"Инициализирован MAPServerMatrixBuilder с параметрами: "
            f"{params.max_customers=}, {params.sensor_count=}, "
            f"{params.lambda_rate=}, {params.mu_rate=}, {params.nu_rate=}"
        )

    def _d_0_matrix_generator(self) -> NDArray[np.float64]:
        """Генерирует матрицу D₀ по MAP-параметрам.

        Returns:
            NDArray[np.float64]: Матрица D₀ для текущих параметров потока.

        Raises:
            ValueError: Если базовые параметры системы не являются MAPSystemParams.
        """
        if not isinstance(self.params, MAPSystemParams):
            logger.error("Базовые параметры не являются экземпляром MAPSystemParams")
            raise ValueError(
                "Базовые параметры должны быть экземпляром MAPSystemParams"
            )

        if self.params.p_rate.shape != self.params.q_rate.shape:
            logger.error("Матрицы p_rate и q_rate должны иметь одинаковую размерность.")
            raise ValueError(
                "Матрицы p_rate и q_rate должны иметь одинаковую размерность."
            )

        logger.debug("Генерация матрицы D₀ из p_rate и λ.")
        matrix = self.params.p_rate.copy()
        matrix *= self.params.lambda_rate[:, np.newaxis]
        np.fill_diagonal(matrix, -self.params.lambda_rate)
        logger.debug(f"Матрица D₀ сгенерирована. shape={matrix.shape}")
        return matrix

    def _d_1_matrix_generator(self) -> NDArray[np.float64]:
        """Генерирует матрицу D₁ по MAP-параметрам.

        Returns:
            NDArray[np.float64]: Матрица D₁ для текущих параметров потока.

        Raises:
            ValueError: Если базовые параметры системы не являются MAPSystemParams.
        """
        if not isinstance(self.params, MAPSystemParams):
            logger.error("Базовые параметры не являются экземпляром MAPSystemParams")
            raise ValueError(
                "Базовые параметры должны быть экземпляром MAPSystemParams"
            )

        logger.debug("Генерация матрицы D₁ из q_rate и λ.")
        matrix = self.params.q_rate.copy()
        matrix *= self.params.lambda_rate[:, np.newaxis]
        logger.debug(f"Матрица D₁ сгенерирована. shape={matrix.shape}")
        return matrix

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
        n, m = self.params.max_customers, self.params.sensor_count
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
