import logging

import numpy as np
from numpy.typing import NDArray

from .parameters import QueueSystemParameters

# Инициализация логгирования
logger = logging.getLogger(__name__)


class CoefficientMatrixBuilder:
    def __init__(self, params: QueueSystemParameters) -> None:
        """
        Инициализация параметров системы.

        :param params: объект QueueSystemParameters с параметрами системы
        """
        self.params = params

    def build(self) -> NDArray[np.float64]:
        """
        Генерация матрицы коэффициентов для системы уравнений.

        :return: матрица коэффициентов для системы уравнений
        """
        n = self.params.max_customers
        λ, μ, ν = self.params.lambda_rate, self.params.mu_rate, self.params.nu_rate

        coefficients_marix = np.zeros((n, n))

        for index in range(n):
            if index == 0:
                # Первое уравнение: dp0/dt = -λp0 + μp1
                # На главной диагонали стоит -λ, а справа (в следующем столбце) стоит μ
                coefficients_marix[index, index] = -λ
                coefficients_marix[index, index+1] = μ
            elif index == n - 1:
                # Последнее уравнение: dpn+1/dt = λpn - (μ + nν)pn+1
                # Ниже главной диагонали стоит λ, а на главной диагонали -(μ + nν)
                coefficients_marix[index, index-1] = λ
                coefficients_marix[index, index] = -(μ + (index - 1) * ν)
            else:
                # Промежуточные уравнения: dpi/dt = λpi-1 - (μ + (i-1)ν + λ)pi + (μ + iν)pi+1
                coefficients_marix[index, index-1] = λ
                coefficients_marix[index, index] = -(μ + (index - 1) * ν + λ)
                coefficients_marix[index, index+1] = μ + index * ν
        
        logger.info("Матрица коэффициентов A для системы уравнений сгенерирована.")
        return coefficients_marix
