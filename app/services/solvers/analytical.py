"""Модуль для аналитического решения системы вероятностей СМО с нетерпеливыми заявками.

Содержит реализации решателей, основанных на NumPy и mpmath, для вычисления динамики
вероятностных состояний системы массового обслуживания (СМО) в переходном режиме.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, cast

import mpmath as mp  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eig as scipy_eig
from tqdm import tqdm  # type: ignore[import-untyped]

from app.db import duckdb_cache
from app.domain import (
    ComputationConfig,
    MergedComputationConfig,
    MpmathComputationConfig,
)
from app.domain.models import SingleServerParams
from app.services.solvers.base import BasicProbabilitySolver

# Инициализация логгирования
logger = logging.getLogger(__name__)


class AnalyticalBasicProbabilitySolver(BasicProbabilitySolver, ABC):
    """Абстрактный базовый класс решателя вероятностей для СМО с нетерпеливыми заявками.

    Определяет общий интерфейс и базовые методы для вычисления вероятностных
    характеристик переходного режима системы массового обслуживания.
    """

    def __init__(
        self,
        params: SingleServerParams,
        coefficients_matrix: NDArray[np.float64],
        config: ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    ) -> None:
        """Инициализирует базовый решатель.

        Args:
            params (SingleServerParams): Параметры системы массового обслуживания.
            coefficients_matrix (NDArray[np.float64]): Матрица коэффициентов системы
                уравнений размером (n x n).
            config: Конфигурация вычислений.
        """
        super().__init__(params, coefficients_matrix, config)

    @abstractmethod
    def generate_m_matrix(
        self,
        eigenvalues: NDArray[np.float64] | Any,
        xsi_matrix: NDArray[np.float64] | Any,
    ) -> NDArray[np.float64]:
        """Генерирует матрицу M(t), зависящую от времени.

        M(t) описывает временную динамику системы и строится на основе собственных
        векторов и значений.

        Args:
            eigenvalues (NDArray[np.float64] | Any): Массив собственных значений
                матрицы коэффициентов размером (n,).
            xsi_matrix (NDArray[np.float64] | Any): Матрица собственных векторов
                размером (n x n).

        Returns:
            NDArray[np.float64]: Трехмерная матрица M(t) размером (n x n x t).
        """
        pass

    @abstractmethod
    def _compute_eigenvalues(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | tuple[Any, Any]:
        """Вычисляет собственные значения и собственные векторы матрицы переходов.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64]] | tuple[Any, Any]:
                Кортеж из массива собственных значений и матрицы собственных векторов.
        """
        pass

    @duckdb_cache("params.initial_probabilities")
    def generate_p_matrix(
        self, m_matrix: NDArray[np.float64 | Any]
    ) -> NDArray[np.float64]:
        """Вычисляет финальную матрицу вероятностей P(t).

        Выполняет умножение начальных вероятностей на матрицу M(t),
        получая вероятности состояний системы в каждый момент времени.

        Args:
            m_matrix (NDArray[np.float64 | Any]): Матрица M размером (n x n x t).

        Returns:
            NDArray[np.float64]: Матрица вероятностей P размером (n x t).
        """
        p_matrix = np.dot(self.params.initial_probabilities, m_matrix).astype(
            np.float64
        )
        logger.info("Вычисление матрицы p завершено.")
        return cast(NDArray[np.float64], p_matrix)

    def calculate(self) -> NDArray[np.float64]:
        """Выполняет полный расчёт вероятностей состояний системы.

        Этапы:
            1. Построение матрицы переходов системы
            2. Вычисление собственных значений матрицы
            3. Решение системы уравнений для вероятностей
            4. Построение итоговой матрицы вероятностей состояний

        Returns:
            NDArray[np.float64]: Матрица вероятностей состояний (размерность
                зависит от параметров СМО).
        """
        eigenvalues, xsi_matrix = self._compute_eigenvalues()
        m_matrix = self.generate_m_matrix(eigenvalues, xsi_matrix)
        return self.generate_p_matrix(m_matrix)


class AnalyticalNumpyProbabilitySolver(AnalyticalBasicProbabilitySolver):
    """Реализация решателя на основе библиотеки NumPy.

    Подходит для стандартных задач с достаточной точностью вычислений.
    """

    def __init__(
        self,
        params: SingleServerParams,
        coefficients_matrix: NDArray[np.float64],
        config: ComputationConfig,
    ) -> None:
        """Инициализирует решатель вероятностей.

        Args:
            params (SingleServerParams): Параметры системы массового обслуживания.
            coefficients_matrix (NDArray[np.float64]): Матрица коэффициентов системы
                уравнений размером (n x n).
            config (ComputationConfig): Конфигурация вычислений.
        """
        super().__init__(params, coefficients_matrix, config)

    @duckdb_cache("coefficients_matrix")
    def _compute_eigenvalues(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Вычисляет собственные значения и собственные векторы матрицы переходов.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64]]:
                Кортеж из массива собственных значений и матрицы собственных векторов.
        """
        eigenvalues, xsi_matrix = scipy_eig(self.coefficients_matrix)

        if not (
            np.all(np.isclose(eigenvalues.imag, 0))
            and np.all(np.isclose(xsi_matrix.imag, 0))
        ):
            raise ValueError(
                "Собственные значения или векторы имеют существенную комплексную часть"
            )

        return (
            np.asarray(eigenvalues.real, dtype=np.float64),
            np.asarray(xsi_matrix.real, dtype=np.float64),
        )

    @duckdb_cache("params.time_array")
    def generate_m_matrix(
        self, eigenvalues: NDArray[np.float64], xsi_matrix: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Генерирует матрицу M, объединяя временные и пространственные характеристики.

        Вычисляет трехмерную матрицу M(t), которая учитывает вклад каждого собственного
        значения и соответствующих векторов в динамику вероятностной системы.

        Args:
            eigenvalues (NDArray[np.float64]): Массив собственных значений матрицы
                коэффициентов размером (n,).
            xsi_matrix (NDArray[np.float64]): Матрица собственных векторов
                размером (n x n).

        Returns:
            NDArray[np.float64]: 3D матрица M размером (n x n x t),
                где t — количество временных точек.
        """
        matrix_size = xsi_matrix.shape[0]
        time_steps = len(self.params.time_array)

        xsi_matrix_inv = np.linalg.inv(xsi_matrix)
        exp_g_t = np.exp(np.outer(eigenvalues, self.params.time_array))

        m_matrix = np.zeros((matrix_size, matrix_size, time_steps), dtype=np.float64)
        for k in tqdm(range(matrix_size), desc="Вычисление слоёв M"):
            outer = np.outer(xsi_matrix[:, k], xsi_matrix_inv[k, :])
            m_matrix += np.real(
                outer[:, :, np.newaxis] * exp_g_t[k, np.newaxis, np.newaxis]
            )

        logger.info("Генерация матрицы M завершена.")
        return m_matrix


class AnalyticalMpmathProbabilitySolver(AnalyticalBasicProbabilitySolver):
    """Реализация решателя с повышенной точностью на основе библиотеки mpmath.

    Предназначена для задач, требующих высокой точности вычислений.
    """

    def __init__(
        self,
        params: SingleServerParams,
        coefficients_matrix: NDArray[np.float64],
        config: MpmathComputationConfig,
    ) -> None:
        """Инициализирует решатель вероятностей с повышенной точностью.

        Args:
            params (SingleServerParams): Параметры системы массового обслуживания.
            coefficients_matrix (NDArray[np.float64]): Матрица коэффициентов системы
                уравнений размером (n x n).
            config (MpmathComputationConfig): Конфигурация вычислений.
        """
        self._precision: int = config.precision
        super().__init__(params, coefficients_matrix, config)

    @duckdb_cache("coefficients_matrix")
    def _compute_eigenvalues(self) -> tuple[Any, Any]:
        """Вычисляет собственные значения и собственные векторы матрицы переходов.

        Returns:
            tuple[Any, Any]: Кортеж из массива собственных значений и матрицы
                собственных векторов.

        Raises:
            ValueError: Если конфиг не является MpmathComputationConfig или
                MergedComputationConfig.
        """
        if not isinstance(
            self.config, MpmathComputationConfig | MergedComputationConfig
        ):
            raise ValueError(
                "Конфиг должн быть экземпляром MpmathComputationConfig или "
                "MergedComputationConfig"
            )

        with mp.workdps(self.config.precision):
            mp_matrix = mp.matrix(self.coefficients_matrix.tolist())
            eigenvalues, xsi_matrix = mp.eig(mp_matrix)
        return eigenvalues, xsi_matrix

    @duckdb_cache("_precision", "params.time_array")
    def _generate_exp_matrix(self, eigenvalues: Any) -> NDArray[Any]:
        """Генерирует матрицу экспонент exp(λ_k * t).

        Внутренний метод для построения матрицы экспоненциального поведения
        собственных значений во времени.

        Args:
            eigenvalues (Any): Массив собственных значений матрицы коэффициентов
                размером (n,).

        Returns:
            NDArray[Any]: Двумерная матрица exp(λ_k * t) для всех λ и t.
        """
        with mp.workdps(self._precision):
            time_array = mp.matrix(self.params.time_array)
            outer = [[eig * t for t in time_array] for eig in eigenvalues]
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
        invalid_indices: NDArray[np.int64] | None = None,
    ) -> NDArray[Any]:
        """Вычисляет матрицу M(t) по слоям с использованием mpmath.

        Args:
            xsi_matrix (Any): Матрица собственных векторов.
            xsi_matrix_inv (Any): Обратная матрица собственных векторов.
            exp_g_t (NDArray[Any]): Матрица экспонент exp(λ_k * t).
            invalid_indices (NDArray[np.int64] | None): Индексы временных точек
                для корректировки вычислений.

        Returns:
            NDArray[Any]: 3D numpy-массив с типом object, содержащий вычисленные
                значения M(t).
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

    def generate_m_matrix(self, eigenvalues: Any, xsi_matrix: Any) -> NDArray[Any]:
        """Генерирует матрицу M(t) с повышенной точностью.

        Вычисляет трехмерную матрицу M(t) на основе собственных векторов и значений,
        используя библиотеку mpmath для обеспечения высокой точности.

        Args:
            eigenvalues (Any): Массив собственных значений матрицы коэффициентов
                размером (n,).
            xsi_matrix (Any): Матрица собственных векторов размером (n x n).

        Returns:
            NDArray[mp.mpf]: 3D матрица M размером (n x n x t),
                где t — количество временных точек.
        """
        with mp.workdps(self._precision):
            xsi_matrix_inv = mp.inverse(xsi_matrix)
            exp_g_t = self._generate_exp_matrix(eigenvalues)
        m_matrix = self._compute_m_matrix(xsi_matrix, xsi_matrix_inv, exp_g_t)
        logger.info("Генерация матрицы M завершена.")
        return m_matrix


class AnalyticalMergedProbabilitySolver(
    AnalyticalMpmathProbabilitySolver, AnalyticalNumpyProbabilitySolver
):
    """Гибридный решатель, сочетающий точность mpmath и производительность numpy."""

    def __init__(
        self,
        params: SingleServerParams,
        coefficients_matrix: NDArray[np.float64],
        config: MergedComputationConfig,
    ) -> None:
        """Инициализирует решатель вероятностей с повышенной точностью.

        Args:
            params (SingleServerParams): Параметры системы массового обслуживания.
            coefficients_matrix (NDArray[np.float64]): Матрица коэффициентов системы
                уравнений размером (n x n).
            config (MergedComputationConfig): Конфигурация вычислений.
        """
        super().__init__(params, coefficients_matrix, config)

    @duckdb_cache("config.tolerance")
    def _get_invalid_indices(
        self, data_matrix: NDArray[np.float64 | Any], check_sum: bool = True
    ) -> NDArray[np.int64]:
        """Определяет индексы временных точек с некорректными значениями.

        Args:
            data_matrix (NDArray[np.float64 | Any]): Трехмерный массив значений M(t).
            check_sum (bool): Флаг проверки суммы по графикам (по умолчанию True).

        Returns:
            NDArray[np.int64]: Массив индексов последних некорректных временных
                точек для каждой пары.
        """
        if not isinstance(self.config, MergedComputationConfig):
            raise ValueError("Конфиг должн быть экземпляром MergedComputationConfig")

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
        sum_over_graphs = data_matrix.sum(axis=0)
        for j in range(matrix_size):
            for t in range(time_steps):
                value_sum = sum_over_graphs[j, t]
                if 1 - self.config.tolerance <= value_sum <= 1 + self.config.tolerance:
                    continue
                invalid_indices[:, j] = np.maximum(invalid_indices[:, j], t)

        return invalid_indices

    def generate_m_matrix(
        self, eigenvalues: Any, xsi_matrix: Any
    ) -> NDArray[np.float64]:
        """Генерирует матрицу M(t) с улучшенной точностью, комбинируя numpy и mpmath.

        Производит начальное вычисление с помощью numpy, затем корректирует
        некорректные значения с использованием mpmath.

        Args:
            eigenvalues (Any): Массив собственных значений матрицы коэффициентов
                размером (n,).
            xsi_matrix (Any): Матрица собственных векторов (n x n).

        Returns:
            NDArray[np.float64]: 3D массив M(t) с исправленными значениями.
        """
        numpy_eigenvalues = np.array(
            [float(mp.re(x)) for x in eigenvalues], dtype=np.float64
        )
        numpy_xsi_matrix = np.array(
            [[float(mp.re(x)) for x in row] for row in xsi_matrix.tolist()],
            dtype=np.float64,
        )
        numpy_m_matrix = AnalyticalNumpyProbabilitySolver.generate_m_matrix(
            self, numpy_eigenvalues, numpy_xsi_matrix
        )
        numpy_invalid_indices = self._get_invalid_indices(numpy_m_matrix)

        if np.all(numpy_invalid_indices == -1):
            logger.info(
                "Корректных значений достаточно, использование numpy достаточно."
            )
            return numpy_m_matrix

        with mp.workdps(self._precision):
            xsi_matrix_inv = mp.inverse(xsi_matrix)
            exp_g_t = self._generate_exp_matrix(eigenvalues)

        # Первичная коррекция неподходящих точек
        mpmath_m_matrix = self._compute_m_matrix(
            xsi_matrix, xsi_matrix_inv, exp_g_t, numpy_invalid_indices
        )
        correction_mask = ~mpmath_m_matrix.astype(bool)
        mpmath_m_matrix[correction_mask] = numpy_m_matrix[correction_mask]

        # Вторичная коррекци для сумм точек и их значений
        mpmath_invalid_indices = self._get_invalid_indices(
            mpmath_m_matrix, check_sum=True
        )
        end_m_matrix = self._compute_m_matrix(
            xsi_matrix, xsi_matrix_inv, exp_g_t, mpmath_invalid_indices
        )
        correction_mask = ~end_m_matrix.astype(bool)
        end_m_matrix[correction_mask] = mpmath_m_matrix[correction_mask]

        logger.info("Коррекция матрицы M(t) с использованием mpmath завершена.")
        return end_m_matrix
