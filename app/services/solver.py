"""Модуль для численного решения системы вероятностей СМО с нетерпеливыми заявками.

Содержит класс ProbabilitySolver, реализующий полный цикл вычисления вероятностных
характеристик системы массового обслуживания в переходном режиме.

Основной функционал:
- Построение матриц M для временного анализа
- Финальное вычисление матрицы вероятностей P(t)
"""

import logging

import numpy as np
from numpy.typing import NDArray

from app.models import SingleServerParams

# Инициализация логгирования
logger = logging.getLogger(__name__)


class ProbabilitySolver:
    """Класс для решения системы вероятностей СМО с нетерпеливыми заявками.

    Осуществляет вычисление вероятностных характеристик системы через:
    1. Генерацию матриц L на основе собственных значений
    2. Расчет значений P для каждого состояния
    3. Построение матриц AA и M
    4. Финальное вычисление матрицы вероятностей P
    """

    def __init__(
        self,
        params: SingleServerParams,
        coefficients_matrix: NDArray[np.float64],
        eigenvalues: NDArray[np.float64],
    ) -> None:
        """Инициализирует решатель вероятностей.

        Аргументы:
            params: Параметры системы массового обслуживания
            coefficients_matrix: Матрица коэффициентов системы уравнений
                                 размером (n x n)
            eigenvalues: Массив собственных значений матрицы коэффициентов размером (n,)
        """
        self.params = params
        self.coefficients_matrix = coefficients_matrix
        self.eigenvalues = eigenvalues

    def generate_m_matrix(self, xsi_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Генерирует матрицу M, объединяя временные и пространственные характеристики.

        Вычисляет трехмерную матрицу M(t), которая учитывает вклад каждого собственного
        значения и соответствующих векторов в динамику вероятностной системы.

        Аргументы:
            xsi_matrix: Матрица собственных векторов размером (n x n)

        Возвращает:
            NDArray[np.float64]: 3D матрица M размером (n x n x t),
                                где t — количество временных точек

        Исключения:
            ValueError: Если размерности входных матриц не согласованы
        """
        matrix_size = xsi_matrix.shape[0]

        xsi_matrix_inv = np.linalg.inv(xsi_matrix)
        exp_g_t = np.exp(np.outer(self.eigenvalues, self.params.time_array))

        m_matrix = np.zeros(
            (matrix_size, matrix_size, len(self.params.time_array)), dtype=np.float64
        )
        for k in range(matrix_size):
            outer = np.outer(xsi_matrix[:, k], xsi_matrix_inv[k, :])
            m_matrix += np.real(
                outer[:, :, np.newaxis] * exp_g_t[k, np.newaxis, np.newaxis]
            )

        logger.info("Генерация матрицы M завершена.")
        return m_matrix

    def generate_p_matrix(self, m_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Вычисляет финальную матрицу вероятностей P.

        Выполняет умножение начальных вероятностей на матрицу M(t),
        получая вероятности состояний системы в каждый момент времени.

        Аргументы:
            m_matrix: Матрица M размером (n x n x t)

        Возвращает:
            NDArray[np.float64]: Матрица вероятностей P размером (n x t)

        Исключения:
            ValueError: Если размерности начальных вероятностей и матрицы M не совпадают
        """
        if self.params.initial_probabilities.shape[0] != m_matrix.shape[0]:
            raise ValueError(
                "Размер initial_probabilities не соответствует размерности m_matrix."
            )

        p_matrix: NDArray[np.float64] = np.dot(
            self.params.initial_probabilities, m_matrix
        )
        logger.info("Вычисление матрицы p завершено.")
        return p_matrix
