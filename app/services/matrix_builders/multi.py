"""Модуль построения матрицы коэффициентов для многоканальной СМО с уходами заявок.

Реализует класс MultiServerMatrixBuilder, который формирует трёхдиагональную матрицу
коэффициентов учитывая количество обслуживающих приборов, интенсивности поступления,
обслуживания и ухода из очереди.
"""

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from app.domain.models import MultiServerParams
from app.services.matrix_builders.base import BaseMatrixBuilder


class MultiServerMatrixBuilder(BaseMatrixBuilder):
    """Построитель матрицы коэффициентов для многоканальной СМО с уходами заявок.

    Учитывает количество обслуживающих приборов, интенсивность обслуживания
    и ухода из очереди.
    """

    def __init__(self, params: MultiServerParams) -> None:
        """Инициализирует построитель многоканальной СМО.

        Args:
            params (MultiServerParams): Параметры многоканальной СМО.
        """
        super().__init__(params)

    def build(self) -> NDArray[np.float64]:
        """Формирует матрицу коэффициентов для многоканальной СМО.

        Returns:
            NDArray[np.float64]: Квадратная матрица коэффициентов
                размера (n+m+1) x (n+m+1).

        Raises:
            ValueError: Если параметры системы не являются MultiServerParams.
        """
        if not isinstance(self.params, MultiServerParams):
            raise ValueError("Параметры должны быть экземпляром MultiServerParams")

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

        logger.info("Матрица коэффициентов для многоканальной СМО сгенерирована.")
        return coefficients_matrix
