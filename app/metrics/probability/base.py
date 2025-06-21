"""Базовый абстрактный класс для расчёта вероятностей в системах массового обслуживания.

Модуль содержит класс BaseProbabilitySystem, обеспечивающий интерфейс и методы
для построения матриц переходов, вычисления собственных значений и решения
системы уравнений переходов для стационарных вероятностей.
"""

from abc import ABC, abstractmethod
from typing import Any

import mpmath as mp  # type: ignore[import-untyped]
import numpy as np
from scipy.linalg import eig as scipy_eig

from app.domain import (
    CalculationEngine,
    ComputationConfig,
    MergedComputationConfig,
    MpmathComputationConfig,
)
from app.domain.models import (
    MAPServerParams,
    MultiServerParams,
    SingleServerParams,
)


class BaseProbabilitySystem(ABC):
    """Базовый абстрактный класс для моделирования СМО с нетерпеливыми заявками.

    Определяет интерфейс и общую логику расчёта вероятностей состояний СМО.
    Наследники реализуют специфичные методы построения матриц переходов.
    """

    def __init__(
        self,
        params: SingleServerParams | MultiServerParams | MAPServerParams,
        config: ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    ) -> None:
        """Инициализирует систему массового обслуживания с заданными параметрами.

        Args:
            params (SingleServerParams | MultiServerParams | MAPServerParams):
                Параметры СМО (интенсивности, структура, др.).
            config: Конфигурация вычислений.
        """
        self.params = params
        self.config = config

    def calculate(self) -> np.ndarray[np.float64]:
        """Выполняет полный расчёт вероятностей состояний системы.

        Этапы:
            1. Построение матрицы переходов системы
            2. Вычисление собственных значений матрицы
            3. Решение системы уравнений для вероятностей
            4. Построение итоговой матрицы вероятностей состояний

        Returns:
            np.ndarray[np.float64]: Матрица вероятностей состояний (размерность
                зависит от параметров СМО).
        """
        transition_matrix = self._build_transition_matrix()
        eigenvalues, xsi_matrix = self._compute_eigenvalues(transition_matrix)
        probability_matrix = self._solve_probability_system(
            transition_matrix, eigenvalues, xsi_matrix
        )
        return probability_matrix

    @abstractmethod
    def _build_transition_matrix(self) -> np.ndarray[np.float64]:
        """Строит матрицу переходов между состояниями СМО.

        Returns:
            np.ndarray[np.float64]: Матрица переходов.
        """
        pass

    def _compute_eigenvalues(
        self, transition_matrix: np.ndarray[np.float64]
    ) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]] | tuple[Any, Any]:
        """Вычисляет собственные значения и собственные векторы матрицы переходов.

        Args:
            transition_matrix (np.ndarray[np.float64]): Матрица переходов.

        Returns:
            tuple[np.ndarray[np.float64], np.ndarray[np.float64]] | tuple[Any, Any]:
                Кортеж из массива собственных значений и матрицы собственных векторов.
        """
        if self.config.calculation_engine in [
            CalculationEngine.MPMATH,
            CalculationEngine.MERGED,
        ] and isinstance(
            self.config, MpmathComputationConfig | MergedComputationConfig
        ):
            with mp.workdps(self.config.precision):
                mp_matrix = mp.matrix(transition_matrix.tolist())
                eigenvalues, xsi_matrix = mp.eig(mp_matrix)
            return eigenvalues, xsi_matrix

        eigenvalues, xsi_matrix = scipy_eig(transition_matrix)
        return eigenvalues, xsi_matrix

    @abstractmethod
    def _solve_probability_system(
        self,
        transition_matrix: np.ndarray[np.float64],
        eigenvalues: np.ndarray[np.float64] | Any,
        xsi_matrix: np.ndarray[np.float64] | Any,
    ) -> np.ndarray[np.float64]:
        """Решает систему уравнений для стационарных вероятностей состояний.

        Args:
            transition_matrix (np.ndarray[np.float64]): Матрица переходов.
            eigenvalues (np.ndarray[np.float64] | Any): Собственные значения.
            xsi_matrix (np.ndarray[np.float64] | Any): Собственные векторы.

        Returns:
            np.ndarray[np.float64]: Матрица стационарных вероятностей.
        """
        pass
