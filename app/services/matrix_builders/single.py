"""Модуль построения матрицы коэффициентов для одноканальной СМО с уходами заявок.

Реализует класс SingleServerMatrixBuilder, который генерирует трёхдиагональную матрицу
коэффициентов для систем массового обслуживания с одной линией и нетерпеливыми заявками,
учитывая интенсивности поступления, обслуживания и ухода клиентов.
"""

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from app.domain.models import MAPSystemParams
from app.services.matrix_builders.base import BaseMatrixBuilder


class SingleServerMatrixBuilder(BaseMatrixBuilder):
    """Построитель матрицы коэффициентов для одноканальной СМО с уходами заявок.

    Генерирует трёхдиагональную матрицу с учётом интенсивностей поступления,
    обслуживания и ухода нетерпеливых клиентов.
    """

    def build(self) -> NDArray[np.float64]:
        """Формирует матрицу коэффициентов для одноканальной СМО.

        Returns:
            NDArray[np.float64]: Квадратная матрица коэффициентов размера n x n.

        Raises:
            ValueError: Если параметры системы являются MAPSystemParams.
        """
        if isinstance(self.params, MAPSystemParams):
            logger.error("Параметры являются экземпляром MAPSystemParams")
            raise ValueError("Параметры не должны быть экземпляром MAPSystemParams")

        n = self.params.max_customers
        λ, μ, ν = self.params.lambda_rate, self.params.mu_rate, self.params.nu_rate

        coefficients_matrix = np.zeros((n, n), dtype=np.float64)

        for index in range(n):
            if index == 0:
                # Начальное состояние
                coefficients_matrix[index, index] = -λ
                coefficients_matrix[index, index + 1] = μ
            elif index == n - 1:
                # Последнее состояние
                coefficients_matrix[index, index - 1] = λ
                coefficients_matrix[index, index] = -(μ + (index - 1) * ν)
            else:
                # Промежуточные состояния
                coefficients_matrix[index, index - 1] = λ
                coefficients_matrix[index, index] = -(μ + (index - 1) * ν + λ)
                coefficients_matrix[index, index + 1] = μ + index * ν

        logger.success(
            "Матрица коэффициентов для одноканальной СМО успешно сгенерирована."
        )
        return coefficients_matrix
