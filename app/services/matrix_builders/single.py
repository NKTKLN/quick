"""Модуль построения матрицы коэффициентов для одноканальной СМО с уходами заявок.

Реализует класс SingleServerMatrixBuilder, который генерирует трёхдиагональную матрицу
коэффициентов для систем массового обслуживания с одной линией и нетерпеливыми заявками,
учитывая интенсивности поступления, обслуживания и ухода клиентов.
"""

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from app.domain.models import SingleServerParams
from app.services.matrix_builders.base import BaseMatrixBuilder


class SingleServerMatrixBuilder(BaseMatrixBuilder):
    """Построитель матрицы коэффициентов для одноканальной СМО с уходами заявок.

    Генерирует трёхдиагональную матрицу с учётом интенсивностей поступления,
    обслуживания и ухода нетерпеливых клиентов.
    """

    def __init__(self, params: SingleServerParams) -> None:
        """Инициализирует построитель одноканальной СМО.

        Args:
            params (SingleServerParams): Параметры одноканальной СМО.
        """
        super().__init__(params)
        logger.debug(
            "Инициализирован SingleServerMatrixBuilder с параметрами: "
            f"max_customers={params.max_customers}, lambda_rate={params.lambda_rate}, "
            f"mu_rate={params.mu_rate}, nu_rate={params.nu_rate}"
        )

    def build(self) -> NDArray[np.float64]:
        """Формирует матрицу коэффициентов для одноканальной СМО.

        Returns:
            NDArray[np.float64]: Квадратная матрица коэффициентов размера n x n.

        Raises:
            ValueError: Если параметры системы не являются SingleServerParams.
        """
        if not isinstance(self.params, SingleServerParams):
            logger.error("Параметры не являются экземпляром SingleServerParams")
            raise ValueError("Параметры должны быть экземпляром SingleServerParams")

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
