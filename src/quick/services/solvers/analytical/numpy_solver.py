"""Модуль аналитического решателя вероятностной системы для СМО через NumPy.

Содержит реализацию класса `AnalyticalNumpyProbabilitySolver`, основанного на NumPy, для
аналитического вычисления динамики вероятностных состояний одноканальных систем
массового обслуживания (СМО) в переходном режиме.
"""

from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.linalg import eig as scipy_eig

from quick.db import duckdb_cache
from quick.domain import ComputationConfig
from quick.domain.params import TransientSystemParams
from quick.services.solvers.analytical.base import AnalyticalBasicProbabilitySolver
from quick.utils import Progress


class AnalyticalNumpyProbabilitySolver(AnalyticalBasicProbabilitySolver):
    """Реализация решателя на основе библиотеки NumPy.

    Подходит для стандартных задач с достаточной точностью вычислений.
    """

    def __init__(
        self,
        params: TransientSystemParams,
        coefficients_matrix: NDArray[np.float64],
        config: ComputationConfig,
    ) -> None:
        """Инициализирует решатель вероятностей.

        Args:
            params (TransientSystemParams): Параметры СМО.
            coefficients_matrix (NDArray[np.float64]): Матрица коэффициентов системы
                уравнений размером (n x n).
            config (ComputationConfig): Конфигурация вычислений.
        """
        super().__init__(params, coefficients_matrix, config)

    @duckdb_cache("coefficients_matrix")
    def _compute_eigenvalues(self) -> tuple[NDArray[Any], NDArray[Any]]:
        """Вычисляет собственные значения и собственные векторы матрицы Q.

        Для MAP-генератора комплексно-сопряжённые собственные значения являются
        штатным случаем. Формулы (8)--(11) статьи допускают такие спектральные
        компоненты: мнимые части взаимно сокращаются в итоговой фундаментальной
        матрице. Поэтому NumPy-решатель больше не отвергает комплексный спектр.

        Returns:
            tuple[NDArray[Any], NDArray[Any]]: Собственные значения и матрица
                собственных векторов в представлении SciPy (обычно complex128).
        """
        logger.debug("Начинаем вычисление собственных значений и векторов...")
        eigenvalues, xsi_matrix = scipy_eig(self.coefficients_matrix)

        if np.any(np.abs(eigenvalues.imag) > 1e-12):
            logger.info(
                "Матрица Q имеет комплексно-сопряжённые собственные значения; "
                "они учитываются в спектральном решении согласно формулам статьи."
            )

        logger.info("Собственные значения и векторы успешно вычислены")
        return np.asarray(eigenvalues), np.asarray(xsi_matrix)

    @duckdb_cache("params.time_array", "cache_revision")
    def _generate_m_matrix_for_last_state(
        self,
        eigenvalues: NDArray[np.float64] | Any,
        xsi_matrix: NDArray[np.float64] | Any,
    ) -> NDArray[np.float64]:
        """Генерирует матрицу M(t) только для последнего состояния системы.

        Используется при конфигурации, когда требуется вычислить вероятности
        состояний только для конечного (последнего) состояния системы,
        а не на всём временном интервале.

        Args:
            eigenvalues (NDArray[np.float64] | Any): Вектор собственных значений
                матрицы переходов размером (n,).
            xsi_matrix (NDArray[np.float64] | Any): Матрица собственных векторов
                размером (n x n).

        Returns:
            NDArray[np.float64]: Матрица M(t) размером (n x n), рассчитанная
                для последнего состояния системы.
        """
        logger.debug("Начинаем генерацию матрицы M(t)...")
        matrix_size = xsi_matrix.shape[0]
        time_steps = len(self.params.time_array)

        xsi_matrix_inv = np.linalg.inv(xsi_matrix)
        elapsed_time = self.params.time_array - self.params.time_array[0]
        exp_g_t = np.exp(np.outer(eigenvalues, elapsed_time))

        m_matrix = np.zeros((matrix_size, matrix_size, time_steps), dtype=np.float64)
        for k in Progress.wrap(range(matrix_size), description="Вычисление слоёв M"):
            outer_row = xsi_matrix[-1, k] * xsi_matrix_inv[k, :]
            m_matrix[-1, :, :] += np.real(np.outer(outer_row, exp_g_t[k]))
            logger.debug(f"Слой {k + 1}/{matrix_size} матрицы M добавлен")

        logger.info("Генерация матрицы M(t) завершена")
        return m_matrix

    @duckdb_cache("params.time_array", "cache_revision")
    def _generate_full_m_matrix(
        self,
        eigenvalues: NDArray[np.float64] | Any,
        xsi_matrix: NDArray[np.float64] | Any,
    ) -> NDArray[np.float64]:
        """Генерирует полную матрицу M(t), зависящую от времени.

        M(t) описывает временную динамику системы и строится на основе собственных
        векторов и значений, отражая вероятности всех состояний системы во все
        моменты времени временного интервала.

        Args:
            eigenvalues (NDArray[np.float64] | Any): Массив собственных значений
                матрицы коэффициентов размером (n,).
            xsi_matrix (NDArray[np.float64] | Any): Матрица собственных векторов
                размером (n x n).

        Returns:
            NDArray[np.float64]: Трехмерная матрица M(t) размером (n x n x t),
                где t - количество временных шагов.
        """
        logger.debug("Начинаем генерацию матрицы M(t)...")
        matrix_size = xsi_matrix.shape[0]
        time_steps = len(self.params.time_array)

        xsi_matrix_inv = np.linalg.inv(xsi_matrix)
        elapsed_time = self.params.time_array - self.params.time_array[0]
        exp_g_t = np.exp(np.outer(eigenvalues, elapsed_time))

        m_matrix = np.zeros((matrix_size, matrix_size, time_steps), dtype=np.float64)
        for k in Progress.wrap(range(matrix_size), description="Вычисление слоёв M"):
            outer = np.outer(xsi_matrix[:, k], xsi_matrix_inv[k, :])
            m_matrix += np.real(
                outer[:, :, np.newaxis] * exp_g_t[k, np.newaxis, np.newaxis]
            )
            logger.debug(f"Слой {k + 1}/{matrix_size} матрицы M добавлен")

        logger.info("Генерация матрицы M(t) завершена")
        return m_matrix

    def generate_m_matrix(
        self,
        eigenvalues: NDArray[np.float64] | Any,
        xsi_matrix: NDArray[np.float64] | Any,
    ) -> NDArray[np.float64]:
        """Генерирует матрицу M(t) в зависимости от конфигурации расчёта.

        Вызывает одну из двух реализаций генерации матрицы:
            - `_generate_m_matrix_for_last_state`: если задан флаг вычисления
            только для последнего состояния системы;
            - `_generate_full_m_matrix`: если нужно получить значения для
            всего временного интервала.

        Args:
            eigenvalues (NDArray[np.float64] | Any): Вектор собственных значений
                матрицы переходов размером (n,).
            xsi_matrix (NDArray[np.float64] | Any): Матрица собственных векторов
                размером (n x n).

        Returns:
            NDArray[np.float64]: Матрица M(t), двух- или трёхмерная, в зависимости
                от типа вычисления.
        """
        if self.config is not None and self.config.compute_only_last_state:
            return self._generate_m_matrix_for_last_state(eigenvalues, xsi_matrix)
        return self._generate_full_m_matrix(eigenvalues, xsi_matrix)
