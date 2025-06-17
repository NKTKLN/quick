"""Модуль для численного решения системы вероятностей СМО с нетерпеливыми заявками.

Содержит реализации решателей, основанных на NumPy и mpmath, для вычисления динамики
вероятностных состояний системы массового обслуживания (СМО) в переходном режиме.

Основные компоненты:
    - BasicProbabilitySolver: абстрактный базовый класс решателя вероятностей.
    - NumpyProbabilitySolver: реализация с использованием numpy.
    - MpmathProbabilitySolver: реализация с использованием mpmath.
    - MergedProbabilitySolver: гибридная реализация.
"""

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Optional, cast

import mpmath as mp  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm  # type: ignore[import-untyped]

from app.db import duckdb_cache
from app.domain import (
    ComputationConfig,
    MergedComputationConfig,
    MpmathComputationConfig,
    SingleServerParams,
)

# Инициализация логгирования
logger = logging.getLogger(__name__)


class BasicProbabilitySolver(ABC):
    """Абстрактный базовый класс решателя вероятностей для СМО с нетерпеливыми заявками.

    Определяет общий интерфейс и базовые методы для вычисления вероятностных
    характеристик переходного режима системы массового обслуживания.
    """

    def __init__(
        self,
        params: SingleServerParams,
        coefficients_matrix: NDArray[np.float64],
        eigenvalues: NDArray[np.float64] | Any,
        config: ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    ) -> None:
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
    ) -> NDArray[np.float64]:
        """Генерирует матрицу M(t), зависящую от времени.

        M(t) описывает временную динамику системы и строится на основе собственных
        векторов и значений.

        Аргументы:
            xsi_matrix: Матрица собственных векторов размером (n x n)

        Возвращает:
            NDArray: Трехмерная матрица M(t) размером (n x n x t)
        """
        pass

    @duckdb_cache("params.initial_probabilities")
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
        return cast(NDArray[np.float64], p_matrix)


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

    @duckdb_cache("params.time_array", "eigenvalues")
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
        time_steps = len(self.params.time_array)

        xsi_matrix_inv = np.linalg.inv(xsi_matrix)
        exp_g_t = np.exp(np.outer(self.eigenvalues, self.params.time_array))

        m_matrix = np.zeros((matrix_size, matrix_size, time_steps), dtype=np.float64)
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
        coefficients_matrix: NDArray[np.float64],
        eigenvalues: Any,
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
        self._precision: int = config.precision
        super().__init__(params, coefficients_matrix, eigenvalues, config)

    @duckdb_cache("_precision", "params.time_array", "eigenvalues")
    def _generate_exp_matrix(self) -> NDArray[Any]:
        """Генерирует матрицу экспонент exp(λ_k * t).

        Внутренний метод для построения матрицы экспоненциального поведения
        собственных значений во времени.

        Возвращает:
            list[list[Any]]: Двумерная матрица exp(λ_k * t) для всех λ и t
        """
        with mp.workdps(self._precision):
            time_array = mp.matrix(self.params.time_array)
            outer = [[eig * t for t in time_array] for eig in self.eigenvalues]
            exp_g_t = np.array(
                [[mp.exp(val) for val in row] for row in outer], dtype=mp.mpf
            )
        return exp_g_t

    @duckdb_cache("params.time_array", "_precision")
    def _compute_m_matrix(
        self,
        xsi_matrix: Any,
        xsi_matrix_inv: Any,
        exp_g_t: NDArray[Any],
        invalid_indices: Optional[NDArray[np.int64]] = None,
    ) -> NDArray[Any]:
        """Вычисляет матрицу M(t) по слоям с использованием mpmath.

        Аргументы:
            xsi_matrix: Матрица собственных векторов.
            xsi_matrix_inv: Обратная матрица собственных векторов.
            exp_g_t: Матрица экспонент exp(λ_k * t).
            invalid_indices: Индексы временных точек для корректировки вычислений.

        Возвращает:
            3D numpy-массив с типом object, содержащий вычисленные значения M(t).
        """
        matrix_size = len(xsi_matrix)
        time_steps = len(self.params.time_array)

        with mp.workdps(self._precision):
            m_matrix = np.zeros((matrix_size, matrix_size, time_steps), dtype=mp.mpf)
            for k in tqdm(range(matrix_size), desc="Пересчёт слоёв M"):
                for i in range(matrix_size):
                    for j in range(matrix_size):
                        time_end_step = (
                            time_steps
                            if invalid_indices is None
                            else invalid_indices[i, j] + 1
                        )
                        if time_end_step == 0:
                            continue

                        outer_ij = xsi_matrix[i, k] * xsi_matrix_inv[k, j]
                        for t in range(time_end_step):
                            m_matrix[i, j, t] += mp.re(outer_ij * exp_g_t[k, t])

            return m_matrix

    def generate_m_matrix(self, xsi_matrix: Any) -> NDArray[Any]:
        """Генерирует матрицу M(t) с повышенной точностью.

        Вычисляет трехмерную матрицу M(t) на основе собственных векторов и значений,
        используя библиотеку mpmath для обеспечения высокой точности.

        Аргументы:
            xsi_matrix: Матрица собственных векторов размером (n x n)

        Возвращает:
            NDArray[mp.mpf]: 3D матрица M размером (n x n x t),
                            где t — количество временных точек
        """
        with mp.workdps(self._precision):
            xsi_matrix_inv = mp.inverse(xsi_matrix)
            exp_g_t = self._generate_exp_matrix()
        m_matrix = self._compute_m_matrix(xsi_matrix, xsi_matrix_inv, exp_g_t)
        logger.info("Генерация матрицы M завершена.")
        return m_matrix


class MergedProbabilitySolver(MpmathProbabilitySolver, NumpyProbabilitySolver):
    """Гибридный решатель, сочетающий точность mpmath и производительность numpy.

    Автоматически переключается между методами для оптимизации вычислений.
    """

    def __init__(
        self,
        params: SingleServerParams,
        coefficients_matrix: NDArray[np.float64],
        eigenvalues: Any,
        config: MergedComputationConfig,
    ) -> None:
        """Инициализирует решатель вероятностей с повышенной точностью.

        Аргументы:
            params: Параметры системы массового обслуживания
            coefficients_matrix: Матрица коэффициентов системы уравнений
                                 размером (n x n)
            eigenvalues: Массив собственных значений матрицы коэффициентов размером (n,)
            config (MergedComputationConfig): Конфигурация вычислений.
        """
        super().__init__(params, coefficients_matrix, eigenvalues, config)

    @duckdb_cache("config.tolerance")
    def _get_invalid_indices(
        self, data_matrix: NDArray[np.float64 | Any], check_sum: bool = True
    ) -> NDArray[np.int64]:
        """Определяет индексы временных точек с некорректными значениями.

        Аргументы:
            data_matrix: Трехмерный массив значений M(t).
            check_sum: Флаг проверки суммы по графикам (по умолчанию True).

        Возвращает:
            Массив индексов последних некорректных временных точек для каждой пары.
        """
        if not isinstance(self.config, MergedComputationConfig):
            raise

        matrix_size = data_matrix.shape[0]
        time_steps = data_matrix.shape[2]

        invalid_indices = np.full((matrix_size, matrix_size), -1, dtype=np.int64)

        # Проверка значений каждого элемента
        for i in range(matrix_size):
            for j in range(matrix_size):
                for t in range(time_steps):
                    value = data_matrix[i, j, t]
                    if -self.config.tolerance <= value <= 1 + self.config.tolerance:
                        continue
                    invalid_indices[i, j] = max(invalid_indices[i, j], t)

        if not check_sum:
            return invalid_indices

        # Проверка суммы по графикам
        sum_over_graphs = data_matrix.sum(axis=(0))
        for j in range(matrix_size):
            for t in range(time_steps):
                value_sum = sum_over_graphs[j, t]
                if 1 - self.config.tolerance <= value_sum <= 1 + self.config.tolerance:
                    continue
                invalid_indices[:, j] = np.maximum(invalid_indices[:, j], t)

        return invalid_indices

    def generate_m_matrix(self, xsi_matrix: Any) -> NDArray[np.float64]:
        """Генерирует матрицу M(t) с улучшенной точностью, комбинируя numpy и mpmath.

        Производит начальное вычисление с помощью numpy, затем корректирует
        некорректные значения с использованием mpmath.

        Аргументы:
            xsi_matrix: Матрица собственных векторов (n x n).

        Возвращает:
            3D массив M(t) с исправленными значениями.
        """
        solver_copy = deepcopy(self)
        solver_copy.eigenvalues = np.array(
            [float(mp.re(x)) for x in self.eigenvalues], dtype=np.float64
        )
        numpy_xsi_matrix = np.array(
            [[float(mp.re(x)) for x in row] for row in xsi_matrix.tolist()],
            dtype=np.float64,
        )

        numpy_m_matrix = NumpyProbabilitySolver.generate_m_matrix(
            solver_copy, numpy_xsi_matrix
        )
        numpy_invalid_indices = self._get_invalid_indices(numpy_m_matrix)

        if np.all(numpy_invalid_indices == -1):
            logger.info(
                "Корректных значений достаточно, использование numpy достаточно."
            )
            return cast(NDArray[np.float64], numpy_m_matrix)

        with mp.workdps(self._precision):
            xsi_matrix_inv = mp.inverse(xsi_matrix)
            exp_g_t = self._generate_exp_matrix()

        # Первичная коррекция неподходящих точек
        mpmath_m_matrix = self._compute_m_matrix(
            xsi_matrix, xsi_matrix_inv, exp_g_t, numpy_invalid_indices
        )
        filter = ~mpmath_m_matrix.astype(bool)
        mpmath_m_matrix[filter] = numpy_m_matrix[filter]

        # Вторичная коррекци для сумм точек и их значений
        mpmath_invalid_indices = self._get_invalid_indices(
            mpmath_m_matrix, check_sum=True
        )
        end_m_matrix = self._compute_m_matrix(
            xsi_matrix, xsi_matrix_inv, exp_g_t, mpmath_invalid_indices
        )
        filter = ~end_m_matrix.astype(bool)
        end_m_matrix[filter] = mpmath_m_matrix[filter]

        logger.info("Коррекция матрицы M(t) с использованием mpmath завершена.")
        return end_m_matrix
