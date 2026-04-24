"""Модуль построения матрицы коэффициентов для многоканальной СМО с уходами заявок.

Реализует класс MultiServerMatrixBuilder, который формирует трёхдиагональную матрицу
коэффициентов учитывая количество обслуживающих приборов, интенсивности поступления,
обслуживания и ухода из очереди.
"""

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.domain.models.system_params import MultiSystemParams
from quick.services.matrix_builders.base import BaseMatrixBuilder


class MultiServerMatrixBuilder(BaseMatrixBuilder):
    """Построитель матрицы коэффициентов для многоканальной СМО с уходами заявок.

    Учитывает количество обслуживающих приборов, интенсивность обслуживания и ухода из
    очереди.
    """

    def __init__(self, params: MultiSystemParams) -> None:
        """Инициализирует базовый построитель с параметрами модели.

        Args:
            params (MultiSystemParams): Параметры модели СМО.
        """
        self.params = params

        logger.debug(
            f"Инициализирован MAPServerMatrixBuilder с параметрами: "
            f"{params.max_customers=}, {params.processor_count=}, "
            f"{params.lambda_rate=}, {params.mu_rate=}, {params.nu_rate=}"
        )

    def build(self) -> NDArray[np.float64]:
        """Формирует матрицу коэффициентов для многоканальной СМО.

        Returns:
            NDArray[np.float64]: Квадратная матрица коэффициентов
                размера (n+m+1) x (n+m+1).

        Raises:
            ValueError: Если параметры системы не являются MultiSystemParams.
        """
        if not isinstance(self.params, MultiSystemParams):
            logger.error("Параметры являются экземпляром MultiSystemParams")
            raise TypeError("Параметры не должны быть экземпляром MultiSystemParams")

        logger.info("Начато построение матрицы коэффициентов.")
        n, m = self.params.max_customers, self.params.processor_count
        λ, μ, ν = self.params.lambda_rate, self.params.mu_rate, self.params.nu_rate

        matrix_size = n + m + 1
        coefficients_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float64)

        for index in range(matrix_size):
            if index == 0:
                # Начальное состояние
                coefficients_matrix[index, index] = -λ
                coefficients_matrix[index, index + 1] = μ
            elif index == matrix_size - 1:
                # Конечное состояние
                coefficients_matrix[index, index - 1] = λ
                coefficients_matrix[index, index] = -(m * μ + n * ν)
            else:
                # Промежуточные состояния
                coefficients_matrix[index, index - 1] = λ
                coefficients_matrix[index, index] = -(
                    λ + min(index, m) * μ + max(0, index - m) * ν
                )
                coefficients_matrix[index, index + 1] = (
                    min(index + 1, m) * μ + max(0, index + 1 - m) * ν
                )

        logger.success(
            "Матрица коэффициентов для многоканальной СМО успешно сгенерирована."
        )
        return coefficients_matrix
