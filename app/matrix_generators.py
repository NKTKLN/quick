"""Модуль для построения матриц коэффициентов систем массового обслуживания.

Модуль содержит классы для генерации матриц коэффициентов систем дифференциальных
уравнений,описывающих поведение одноканальных и многоканальных СМО с нетерпеливыми
заявками.

Основные классы:
1. QueueSystemMatrixBuilder - построитель матрицы для одноканальной СМО
2. MultiQueueSystemMatrixBuilder - построитель матрицы для многоканальной СМО

Основные функции:
- Построение трехдиагональных матриц коэффициентов
- Учет параметров системы: интенсивностей входящего потока, обслуживания и ухода
- Поддержка систем с ограниченной очередью
- Генерация матриц как для одноканальных, так и для многоканальных систем
"""

import logging

import numpy as np
from numpy.typing import NDArray

from app.parameters import MultiQueueSystemParameters, QueueSystemParameters

# Инициализация логгирования
logger = logging.getLogger(__name__)


class QueueSystemMatrixBuilder:
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

        coefficients_matrix = np.zeros((n, n), dtype=np.float64)

        for index in range(n):
            if index == 0:
                # Первое уравнение:
                #  dp0/dt = -λp0 + μp1
                # На главной диагонали стоит -λ, а справа (в следующем столбце) стоит μ
                coefficients_matrix[index, index] = -λ
                coefficients_matrix[index, index + 1] = μ
            elif index == n - 1:
                # Последнее уравнение:
                #  dpn+1/dt = λpn - (μ + nν)pn+1
                # Ниже главной диагонали стоит λ, а на главной диагонали -(μ + nν)
                coefficients_matrix[index, index - 1] = λ
                coefficients_matrix[index, index] = -(μ + (index - 1) * ν)
            else:
                # Промежуточные уравнения:
                #  dpi/dt = λpi-1 - (μ + (i-1)ν + λ)pi + (μ + iν)pi+1
                coefficients_matrix[index, index - 1] = λ
                coefficients_matrix[index, index] = -(μ + (index - 1) * ν + λ)
                coefficients_matrix[index, index + 1] = μ + index * ν

        logger.info("Матрица коэффициентов A для системы уравнений сгенерирована.")
        return coefficients_matrix


class MultiQueueSystemMatrixBuilder(QueueSystemMatrixBuilder):
    """Класс для построения матрицы коэффициентов многоканальной системы СМО.

    Наследует базовый функционал от QueueSystemMatrixBuilder и расширяет его
    для случая многоканальных систем с несколькими обслуживающими приборами.
    Строит трехдиагональную матрицу коэффициентов с учетом количества процессоров.
    """

    def __init__(self, params: MultiQueueSystemParameters) -> None:
        """Инициализирует построитель матрицы с параметрами системы.

        Args:
            params: Параметры системы массового обслуживания, включая:
                    - lambda_rate (λ): интенсивность входящего потока
                    - mu_rate (μ): интенсивность обслуживания
                    - nu_rate (ν): интенсивность ухода заявки из очереди
                    - max_customers (n): максимальное число заявок в системе
                    - processor_count (m): количество обслуживающих приборов в системе
        """
        self.params = params

    def build(self) -> NDArray[np.float64]:
        """Строит матрицу коэффициентов для многоканальной системы.

        Матрица строится с учетом количества обслуживающих приборов (m):
        - Для состояний, где число заявок меньше m: все приборы активны
        - Для состояний, где число заявок больше m: работают только m приборов,
          остальные заявки ждут в очереди и могут уйти с интенсивностью ν

        Returns:
            NDArray[np.float64]: Квадратная матрица коэффициентов размером
                                 (n+m+1) x (n+m+1), где n - максимальное число заявок
                                 в системе, m - количество обслуживающих приборов.
                                 Матрица имеет трехдиагональную структуру.
        """
        n, m = self.params.max_customers, self.params.processor_count
        λ, μ, ν = self.params.lambda_rate, self.params.mu_rate, self.params.nu_rate

        matrix_size = n + m + 1
        coefficients_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float64)

        for index in range(matrix_size):
            if index == 0:
                # Первое уравнение:
                #  dp0/dt = -λp0 + μp1
                # На главной диагонали стоит -λ, а справа (в следующем столбце) стоит μ
                coefficients_matrix[index, index] = -λ
                coefficients_matrix[index, index + 1] = μ
            elif index == matrix_size - 1:
                # Последнее уравнение:
                #  dpn+1/dt = λpn - (λ + m * μ + nν)pn+1
                # Ниже главной диагонали стоит λ,
                # а на главной диагонали -(λ + m * μ + nν)
                coefficients_matrix[index, index - 1] = λ
                coefficients_matrix[index, index] = -(m * μ + n * ν)
            else:
                # Промежуточные уравнения:
                #  dpi/dt = λ·pi-1 - [μ·min(i,m) + ν·max(0,i-m) + λ]·pi +
                #           [μ·min(i+1,m) + ν·max(0,i+1-m)]·pi+1
                coefficients_matrix[index, index - 1] = λ
                coefficients_matrix[index, index] = -(λ + m * μ + (index - m) * ν)

                if index + 1 <= m:
                    coefficients_matrix[index, index + 1] = (index + 1) * μ
                else:
                    coefficients_matrix[index, index + 1] = m * μ + (index + 1 - m) * ν

        logger.info(
            "Матрица коэффициентов A для многолинейной системы уравнений сгенерирована."
        )
        return coefficients_matrix
