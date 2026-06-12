"""Модуль для аналитического решения системы вероятностей СМО через mpmath.

Содержит реализацию класса `AnalyticalMpmathProbabilitySolver`, который использует
библиотеку mpmath для вычисления собственных значений и векторов, генерации матрицы
экспонент и матрицы переходов M(t) с целью получения более точных результатов в
моделировании динамики вероятностных состояний одноканальной системы массового
обслуживания (СМО) в переходном режиме.
"""

from typing import Any

import mpmath as mp  # type: ignore[import-untyped]
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.db import duckdb_cache
from quick.domain import (
    MergedComputationConfig,
    MpmathComputationConfig,
)
from quick.domain.params import TransientSystemParams
from quick.utils import Progress

from .base import AnalyticalBasicProbabilitySolver


class AnalyticalMpmathProbabilitySolver(AnalyticalBasicProbabilitySolver):
    """Реализация решателя с повышенной точностью на основе библиотеки mpmath.

    Предназначена для задач, требующих высокой точности вычислений.
    """

    def __init__(
        self,
        params: TransientSystemParams,
        coefficients_matrix: NDArray[np.float64],
        config: MpmathComputationConfig,
    ) -> None:
        """Инициализирует решатель вероятностей с повышенной точностью.

        Args:
            params (TransientSystemParams): Параметры СМО.
            coefficients_matrix (NDArray[np.float64]): Матрица коэффициентов системы
                уравнений размером (n x n).
            config (MpmathComputationConfig): Конфигурация вычислений.
        """
        super().__init__(params, coefficients_matrix, config)
        self._precision = config.precision

    @duckdb_cache("coefficients_matrix")
    def _compute_eigenvalues(self) -> tuple[Any, Any]:
        """Вычисляет собственные значения и собственные векторы матрицы переходов.

        Returns:
            tuple[Any, Any]: Кортеж из массива собственных значений и матрицы
                собственных векторов.

        Raises:
            TypeError: Если конфиг не является MpmathComputationConfig или
                MergedComputationConfig.
        """
        logger.debug("Начинаем вычисление собственных значений и векторов...")
        if not isinstance(
            self.config, MpmathComputationConfig | MergedComputationConfig
        ):
            logger.error("Конфиг неверного типа для mpmath решателя.")
            raise TypeError(
                "Конфиг должн быть экземпляром MpmathComputationConfig или "
                "MergedComputationConfig"
            )

        with mp.workdps(self.config.precision):
            mp_matrix = mp.matrix(self.coefficients_matrix.tolist())
            eigenvalues, xsi_matrix = mp.eig(mp_matrix)

        logger.info("Собственные значения и векторы успешно вычислены")
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
        logger.debug("Начинаем генерацию матрицы экспонент...")
        with mp.workdps(self._precision):
            time_array = mp.matrix(self.params.time_array)
            outer = [[eig * t for t in time_array] for eig in eigenvalues]
            exp_g_t = np.array(
                [[mp.exp(val) for val in row] for row in outer], dtype=mp.mpf
            )
        logger.info("Матрица экспонент успешно сгенерирована")
        return exp_g_t

    @duckdb_cache("params.time_array")
    def _generate_m_matrix_for_last_state(
        self,
        xsi_matrix: Any,
        xsi_matrix_inv: Any,
        exp_g_t: NDArray[Any],
        invalid_indices: NDArray[np.int64] | None = None,
    ) -> NDArray[np.float64]:
        """Генерирует матрицу M(t) только для последнего состояния системы.

        Используется при конфигурации, когда требуется вычислить вероятности
        состояний только для конечного (последнего) состояния системы,
        а не на всём временном интервале.

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
        logger.debug("Начинаем вычисление матрицы M(t)...")
        matrix_size = len(xsi_matrix)
        last_index = matrix_size - 1
        time_steps = len(self.params.time_array)

        with mp.workdps(self._precision):
            m_matrix = np.zeros((matrix_size, matrix_size, time_steps), dtype=mp.mpf)
            for k in Progress.wrap(range(matrix_size), description="Пересчёт слоёв M"):
                for j in range(matrix_size):
                    time_end_step = (
                        time_steps
                        if invalid_indices is None
                        else invalid_indices[last_index, j] + 1
                    )
                    if time_end_step == 0:
                        continue

                    outer_ij = xsi_matrix[last_index, k] * xsi_matrix_inv[k, j]
                    for t in range(time_end_step):
                        m_matrix[last_index, j, t] += mp.re(outer_ij * exp_g_t[k, t])
                logger.debug(f"Слой {k + 1}/{matrix_size} матрицы M добавлен")
            logger.debug("Вычисление матрицы M(t) завершено")
            return m_matrix

    @duckdb_cache("params.time_array")
    def _generate_full_m_matrix(
        self,
        xsi_matrix: Any,
        xsi_matrix_inv: Any,
        exp_g_t: NDArray[Any],
        invalid_indices: NDArray[np.int64] | None = None,
    ) -> NDArray[np.float64]:
        """Генерирует полную матрицу M(t), зависящую от времени.

        M(t) описывает временную динамику системы и строится на основе собственных
        векторов и значений, отражая вероятности всех состояний системы во все
        моменты времени временного интервала.

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
        logger.debug("Начинаем вычисление матрицы M(t)...")
        matrix_size = len(xsi_matrix)
        time_steps = len(self.params.time_array)

        with mp.workdps(self._precision):
            m_matrix = np.zeros((matrix_size, matrix_size, time_steps), dtype=mp.mpf)
            for k in Progress.wrap(range(matrix_size), description="Пересчёт слоёв M"):
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
                logger.debug(f"Слой {k + 1}/{matrix_size} матрицы M добавлен")

            logger.debug("Вычисление матрицы M(t) завершено")
            return m_matrix

    def _generate_m_matrix(
        self,
        xsi_matrix: Any,
        xsi_matrix_inv: Any,
        exp_g_t: NDArray[Any],
        invalid_indices: NDArray[np.int64] | None = None,
    ) -> NDArray[np.float64]:
        """Генерирует матрицу M(t) в зависимости от конфигурации расчёта.

        Вызывает одну из двух реализаций генерации матрицы:
            - `_generate_m_matrix_for_last_state`: если задан флаг вычисления
            только для последнего состояния системы;
            - `_generate_full_m_matrix`: если нужно получить значения для
            всего временного интервала.

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
        if self.config is not None and self.config.compute_only_last_state:
            return self._generate_m_matrix_for_last_state(
                xsi_matrix, xsi_matrix_inv, exp_g_t, invalid_indices
            )
        return self._generate_full_m_matrix(
            xsi_matrix, xsi_matrix_inv, exp_g_t, invalid_indices
        )

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
                где t - количество временных точек.
        """
        logger.debug("Начинаем генерацию матрицы M(t)...")
        with mp.workdps(self._precision):
            xsi_matrix_inv = mp.inverse(xsi_matrix)
            exp_g_t = self._generate_exp_matrix(eigenvalues)
        m_matrix = self._generate_m_matrix(xsi_matrix, xsi_matrix_inv, exp_g_t)
        logger.info("Генерация матрицы M(t) завершена")
        return m_matrix
