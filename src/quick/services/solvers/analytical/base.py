"""Базовый модуль аналитического решателя вероятностной системы для СМО с уходом заявок.

Содержит абстрактный класс `AnalyticalBasicProbabilitySolver`, который определяет
интерфейс и реализует базовые методы для аналитического вычисления вероятностных
характеристик переходного режима одноканальных СМО с нетерпеливыми заявками.
"""

from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.db import duckdb_cache
from quick.domain import (
    ComputationConfig,
)
from quick.domain.models import TransientSystemParams
from quick.services.solvers.base import BasicProbabilitySolver


class AnalyticalBasicProbabilitySolver(BasicProbabilitySolver, ABC):
    """Абстрактный базовый класс решателя вероятностей для СМО с нетерпеливыми заявками.

    Определяет общий интерфейс и базовые методы для вычисления вероятностных
    характеристик переходного режима системы массового обслуживания.
    """

    def __init__(
        self,
        params: TransientSystemParams,
        coefficients_matrix: NDArray[np.float64],
        config: type[ComputationConfig],
    ) -> None:
        """Инициализирует базовый решатель.

        Args:
            params (TransientSystemParams): Параметры СМО.
            coefficients_matrix (NDArray[np.float64]): Матрица коэффициентов системы
                уравнений размером (n x n).
            config: Конфигурация вычислений.
        """
        super().__init__(params, coefficients_matrix, config)

    @abstractmethod
    def _compute_eigenvalues(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | tuple[Any, Any]:
        """Вычисляет собственные значения и собственные векторы матрицы переходов.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64]] | tuple[Any, Any]:
                Кортеж из массива собственных значений и матрицы собственных векторов.
        """

    @abstractmethod
    def generate_m_matrix(
        self,
        eigenvalues: NDArray[np.float64] | Any,
        xsi_matrix: NDArray[np.float64] | Any,
    ) -> NDArray[np.float64]:
        """Генерирует матрицу M(t).

        Args:
            eigenvalues (NDArray[np.float64] | Any): Вектор собственных значений
                матрицы переходов размером (n,).
            xsi_matrix (NDArray[np.float64] | Any): Матрица собственных векторов
                размером (n x n).

        Returns:
            NDArray[np.float64]: Матрица M(t), двух- или трёхмерная, в зависимости
                от типа вычисления.
        """

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
        logger.debug("Вычисляем итоговую матрицу вероятностей P(t)...")
        p_matrix = np.dot(self.params.initial_probabilities, m_matrix).astype(
            np.float64
        )
        logger.info("Вычисление матрицы P(t) завершено успешно")
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
        logger.info("Запуск полного расчёта вероятностей состояния системы...")
        eigenvalues, xsi_matrix = self._compute_eigenvalues()
        m_matrix = self.generate_m_matrix(eigenvalues, xsi_matrix)
        p_matrix = self.generate_p_matrix(m_matrix)
        logger.success("Полный расчёт вероятностей завершён успешно")
        return p_matrix
