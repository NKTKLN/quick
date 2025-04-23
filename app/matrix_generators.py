"""Модуль генерации коэффициентной матрицы для дифференциальной модели СМО.

Содержит класс CoefficientMatrixBuilder, который создает матрицу коэффициентов
для системы дифференциальных уравнений, описывающих поведение СМО с учетом
входящего потока, обслуживания и ухода заявок.
"""

import logging

import numpy as np
from numpy.typing import NDArray

from app.parameters import QueueSystemParameters

# Инициализация логгирования
logger = logging.getLogger(__name__)


class CoefficientMatrixBuilder:
    """Класс для построения матрицы коэффициентов системы СМО.

    Строит трехдиагональную матрицу коэффициентов для системы дифференциальных
    уравнений, описывающих поведение системы массового обслуживания с
    нетерпеливыми заявками.
    """

    def __init__(self, params: QueueSystemParameters) -> None:
        """Инициализирует построитель матрицы с параметрами системы.

        Args:
            params: Параметры системы массового обслуживания, включая:
                   - lambda_rate (λ): интенсивность входящего потока
                   - mu_rate (μ): интенсивность обслуживания
                   - nu_rate (ν): интенсивность ухода заявки из очереди
                   - max_customers (n): максимальное число заявок в системе
        """
        self.params = params

    def build(self) -> NDArray[np.float64]:
        """Строит матрицу коэффициентов системы дифференциальных уравнений.

        Матрица строится по следующим правилам:
        - Для первого уравнения (i=0): коэффициенты -λ (диагональ) и μ (над диагональю)
        - Для последнего уравнения (i=n-1): коэффициенты λ (под диагональю)
          и -(μ+(n-1)ν) (диагональ)
        - Для промежуточных уравнений: тридиагональная структура с коэффициентами:
          λ (под диагональю), -(μ+(i-1)ν+λ) (диагональ), μ+iν (над диагональю)

        Returns:
            NDArray[np.float64]: Квадратная матрица коэффициентов размером n x n, где
                               n - максимальное число заявок в системе.
                               Матрица имеет трехдиагональную структуру.
        """
        n = self.params.max_customers
        λ, μ, ν = self.params.lambda_rate, self.params.mu_rate, self.params.nu_rate

        coefficients_marix = np.zeros((n, n))

        for index in range(n):
            if index == 0:
                # Первое уравнение:
                #  dp0/dt = -λp0 + μp1
                # На главной диагонали стоит -λ, а справа (в следующем столбце) стоит μ
                coefficients_marix[index, index] = -λ
                coefficients_marix[index, index + 1] = μ
            elif index == n - 1:
                # Последнее уравнение:
                #  dpn+1/dt = λpn - (μ + nν)pn+1
                # Ниже главной диагонали стоит λ, а на главной диагонали -(μ + nν)
                coefficients_marix[index, index - 1] = λ
                coefficients_marix[index, index] = -(μ + (index - 1) * ν)
            else:
                # Промежуточные уравнения:
                #  dpi/dt = λpi-1 - (μ + (i-1)ν + λ)pi + (μ + iν)pi+1
                coefficients_marix[index, index - 1] = λ
                coefficients_marix[index, index] = -(μ + (index - 1) * ν + λ)
                coefficients_marix[index, index + 1] = μ + index * ν

        logger.info("Матрица коэффициентов A для системы уравнений сгенерирована.")
        return coefficients_marix
