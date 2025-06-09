"""Модуль для численного решения системы вероятностей СМО с нетерпеливыми заявками.

Содержит реализацию решателей, основанных на NumPy и mpmath, для вычисления динамики
вероятностных состояний системы массового обслуживания (СМО) в переходном режиме.

Основные компоненты:
    - BasicProbabilitySolver: абстрактный базовый класс решателя
    - NumpyProbabilitySolver: реализация через numpy для стандартной точности
    - MpmathProbabilitySolver: реализация через mpmath для повышенной точности
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import mpmath as mp
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from app.models import (
    ComputationConfig,
    MergedComputationConfig,
    MpmathComputationConfig,
    SingleServerParams,
)

# Инициализация логгирования
logger = logging.getLogger(__name__)


class BasicProbabilitySolver(ABC):
    """Базовый абстрактный класс решателя вероятностей для СМО.

    Предоставляет общую структуру и интерфейс для вычисления вероятностей.
    """

    def __init__(
        self,
        params: SingleServerParams,
        coefficients_matrix,
        eigenvalues,
        config: ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    ):
        """Инициализирует базовый решатель.

        Аргументы:
            params: Параметры системы массового обслуживания.
            coefficients_matrix: Матрица коэффициентов системы уравнений (n x n).
            eigenvalues: Массив собственных значений матрицы коэффициентов (n,).
            config (ComputationConfig): Конфигурация вычислений.
        """
        self.params = params
        self.coefficients_matrix = coefficients_matrix
        self.eigenvalues = eigenvalues
        self.config = config

    @abstractmethod
    def generate_m_matrix(
        self, xsi_matrix: NDArray[np.float64] | Any
    ) -> NDArray[np.float64 | Any]:
        """Генерирует матрицу M(t), зависящую от времени.

        M(t) описывает временную динамику системы и строится на основе собственных
        векторов и значений.

        Аргументы:
            xsi_matrix: Матрица собственных векторов размером (n x n)

        Возвращает:
            NDArray: Трехмерная матрица M(t) размером (n x n x t)
        """
        pass

    def generate_p_matrix(
        self, m_matrix: NDArray[np.float64 | Any]
    ) -> NDArray[np.float64]:
        """Вычисляет финальную матрицу вероятностей P(t).

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

        p_matrix = np.dot(self.params.initial_probabilities, m_matrix).astype(
            np.float64
        )
        logger.info("Вычисление матрицы p завершено.")
        return p_matrix


class NumpyProbabilitySolver(BasicProbabilitySolver):
    """Реализация решателя на основе библиотеки NumPy.

    Подходит для стандартных задач с достаточной точностью вычислений.
    """

    def __init__(
        self,
        params: SingleServerParams,
        coefficients_matrix: NDArray[np.float64],
        eigenvalues: NDArray[np.float64],
        config: ComputationConfig,
    ) -> None:
        """Инициализирует решатель вероятностей.

        Аргументы:
            params: Параметры системы массового обслуживания
            coefficients_matrix: Матрица коэффициентов системы уравнений
                                 размером (n x n)
            eigenvalues: Массив собственных значений матрицы коэффициентов размером (n,)
            config (ComputationConfig): Конфигурация вычислений.
        """
        super().__init__(params, coefficients_matrix, eigenvalues, config)

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
        time_array_size = len(self.params.time_array)

        xsi_matrix_inv = np.linalg.inv(xsi_matrix)
        exp_g_t = np.exp(np.outer(self.eigenvalues, self.params.time_array))

        m_matrix = np.zeros(
            (matrix_size, matrix_size, time_array_size), dtype=np.float64
        )
        for k in tqdm(range(matrix_size), desc="Вычисление слоёв M"):
            outer = np.outer(xsi_matrix[:, k], xsi_matrix_inv[k, :])
            m_matrix += np.real(
                outer[:, :, np.newaxis] * exp_g_t[k, np.newaxis, np.newaxis]
            )

        logger.info("Генерация матрицы M завершена.")
        return m_matrix


class MpmathProbabilitySolver(BasicProbabilitySolver):
    """Реализация решателя с повышенной точностью на основе библиотеки mpmath.

    Предназначена для задач, требующих высокой точности вычислений.
    """

    def __init__(
        self,
        params: SingleServerParams,
        coefficients_matrix,
        eigenvalues,
        config: MpmathComputationConfig,
    ) -> None:
        """Инициализирует решатель вероятностей с повышенной точностью.

        Аргументы:
            params: Параметры системы массового обслуживания
            coefficients_matrix: Матрица коэффициентов системы уравнений
                                 размером (n x n)
            eigenvalues: Массив собственных значений матрицы коэффициентов размером (n,)
            config (MpmathComputationConfig): Конфигурация вычислений.
        """
        self.precision = config.precision
        super().__init__(params, coefficients_matrix, eigenvalues, config)

    def _generate_exp_matrix(self):
        """Генерирует матрицу экспонент exp(λ_k * t).

        Внутренний метод для построения матрицы экспоненциального поведения
        собственных значений во времени.

        Возвращает:
            list[list[mp.mpf]]: Двумерная матрица exp(λ_k * t) для всех λ и t
        """
        with mp.workdps(self.precision):
            time_array = mp.matrix(self.params.time_array)
            outer = [[eig * t for t in time_array] for eig in self.eigenvalues]
            exp_g_t = [[mp.exp(val) for val in row] for row in outer]
        return exp_g_t

    def generate_m_matrix(self, xsi_matrix):
        """Генерирует матрицу M(t) с повышенной точностью.

        Вычисляет трехмерную матрицу M(t) на основе собственных векторов и значений,
        используя библиотеку mpmath для обеспечения высокой точности.

        Аргументы:
            xsi_matrix: Матрица собственных векторов размером (n x n)

        Возвращает:
            NDArray[mp.mpf]: 3D матрица M размером (n x n x t),
                            где t — количество временных точек
        """
        matrix_size = len(xsi_matrix)
        time_array_size = len(self.params.time_array)

        with mp.workdps(self.precision):
            xsi_matrix_inv = mp.inverse(xsi_matrix)
            exp_g_t = self._generate_exp_matrix()

            m_matrix = np.zeros(
                (matrix_size, matrix_size, time_array_size), dtype=mp.mpf
            )
            for k in tqdm(range(matrix_size), desc="Вычисление слоёв M"):
                for i in range(matrix_size):
                    for j in range(matrix_size):
                        outer_ij = xsi_matrix[i, k] * xsi_matrix_inv[k, j]
                        for t in range(time_array_size):
                            m_matrix[i][j][t] += mp.re(outer_ij * exp_g_t[k][t])

        logger.info("Генерация матрицы M завершена.")
        return m_matrix
