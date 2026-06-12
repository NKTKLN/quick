"""Модуль построения матрицы коэффициентов для СМО с Марковскими входными потоками и несколькими датчиками.

Содержит класс MultiSensorMAPServerMatrixBuilder, который строит матрицу коэффициентов
систем массового обслуживания с несколькими датчикаминесколькими датчиками, в которых
входной поток описывается марковским процессом (MAP). Основан на параметрах переходов
между состояниями MAP и интенсивностях обслуживания и ухода.
"""

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.domain.params import MAPSystemParams
from quick.services.matrix_builders.base import BaseMatrixBuilder
from quick.services.matrix_builders.utils import (
    map_d_0_matrix_generator,
    map_d_1_matrix_generator,
)


class MultiSensorMAPServerMatrixBuilder(BaseMatrixBuilder):
    """Построитель матрицы коэффициентов для СМО с Марковскими входными потоками и несколькими датчиками.

    Использует матрицы вероятностей переходов p_rate и q_rate для построения
    четырёхмерной матрицы, сворачиваемой в двухмерную.
    """

    def __init__(self, params: MAPSystemParams, debug_logs: bool = False) -> None:
        """Инициализирует базовый построитель с параметрами модели.

        Args:
            params (MAPSystemParams): Параметры модели СМО.
            debug_logs (bool): Логгировать INFO и SUCCESS в DEBUG.
        """
        self.params = params

        self.log_info = logger.debug if debug_logs else logger.info
        self.log_success = logger.debug if debug_logs else logger.success

        logger.debug(
            f"Инициализирован MAPServerMatrixBuilder с параметрами: "
            f"{params.max_customers=}, {params.sensor_count=}, "
            f"{params.lambda_rate=}, {params.mu_rate=}, {params.nu_rate=}"
        )

    def build(self) -> NDArray[np.float64]:
        """Формирует матрицу коэффициентов для СМО с Марковскими входными потоками.

        Returns:
            NDArray[np.float64]: Квадратная матрица коэффициентов.

        Raises:
            TypeError: Если базовые параметры системы не являются MAPSystemParams.
            ValueError: Если матрицы p_rate и q_rate имеют разную размерность.
        """
        if not isinstance(self.params, MAPSystemParams):
            logger.error("Базовые параметры не являются экземпляром MAPSystemParams")
            raise TypeError("Базовые параметры должны быть экземпляром MAPSystemParams")

        if self.params.p_rate.shape != self.params.q_rate.shape:
            logger.error("Матрицы p_rate и q_rate должны иметь одинаковую размерность.")
            raise ValueError(
                "Матрицы p_rate и q_rate должны иметь одинаковую размерность."
            )

        self.log_info("Начато построение матрицы коэффициентов.")
        n, m = self.params.max_customers, self.params.sensor_count
        μ, ν = self.params.mu_rate, self.params.nu_rate

        d_0_t = map_d_0_matrix_generator(self.params.p_rate, self.params.lambda_rate).T
        d_1_t = map_d_1_matrix_generator(self.params.q_rate, self.params.lambda_rate).T

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

        self.log_success(
            "Матрица коэффициентов для СМО с Марковскими входными потоками и несколькими датчикам "
            "успешно сгенерирована."
        )
        return coefficients_matrix
