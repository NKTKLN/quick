"""Модуль для моделирования систем массового обслуживания с нетерпеливыми заявками.

Включает классы для вычисления вероятностных характеристик одно- и многолинейных СМО,
в том числе с MAP-потоками поступления. Используются матричные методы. Поддерживаются
вычисления с обычной точностью (numpy) и повышенной точностью (mpmath).
"""

import logging
from typing import Any

import numpy as np

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
from app.metrics.probability.base import BaseProbabilitySystem
from app.services.matrix_generators import (
    MAPServerMatrixBuilder,
    MultiServerMatrixBuilder,
    SingleServerMatrixBuilder,
)
from app.services.solvers.analytical import (
    MergedProbabilitySolver,
    MpmathProbabilitySolver,
    NumpyProbabilitySolver,
)

# Настройка логирования для отслеживания работы системы
logger = logging.getLogger(__name__)


class BaseAnalyticalProbabilitySystem(BaseProbabilitySystem):
    """Базовый класс для аналитического моделирования СМО с нетерпеливыми заявками.

    Определяет интерфейс и общую логику расчёта вероятностей состояний СМО.
    Наследники реализуют специфичные методы построения матриц переходов.
    """

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
        prob_solver: (
            NumpyProbabilitySolver | MpmathProbabilitySolver | MergedProbabilitySolver
        )
        match self.config.calculation_engine:
            case CalculationEngine.NUMPY if isinstance(self.config, ComputationConfig):
                prob_solver = NumpyProbabilitySolver(
                    self.params, transition_matrix, eigenvalues, self.config
                )
            case CalculationEngine.MPMATH if isinstance(
                self.config, MpmathComputationConfig
            ):
                prob_solver = MpmathProbabilitySolver(
                    self.params, transition_matrix, eigenvalues, self.config
                )
            case CalculationEngine.MERGED if isinstance(
                self.config, MergedComputationConfig
            ):
                prob_solver = MergedProbabilitySolver(
                    self.params, transition_matrix, eigenvalues, self.config
                )
            case _:
                raise ValueError("Неподдерживаемый тип расчета")

        m_matrix = prob_solver.generate_m_matrix(xsi_matrix)
        return prob_solver.generate_p_matrix(m_matrix)


class AnalyticalSingleServerSystem(BaseAnalyticalProbabilitySystem):
    """Класс моделирования однолинейной СМО с нетерпеливыми заявками.

    Реализует методы построения матрицы переходов и расчёта вероятностей
    для системы с одним каналом обслуживания.
    """

    def __init__(
        self,
        params: SingleServerParams,
        config: ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    ) -> None:
        """Инициализирует однолинейную систему массового обслуживания.

        Args:
            params (SingleServerParams): Параметры системы.
            config: Конфигурация вычислений.
        """
        super().__init__(params, config)

    def _build_transition_matrix(self) -> np.ndarray[np.float64]:
        """Строит матрицу переходов однолинейной системы.

        Returns:
            np.ndarray[np.float64]: Матрица переходов между состояниями системы.
        """
        transition_matrix = SingleServerMatrixBuilder(self.params).build()
        return transition_matrix


class AnalyticalMultiServerSystem(BaseAnalyticalProbabilitySystem):
    """Класс моделирования многолинейной СМО с нетерпеливыми заявками.

    Реализует методы построения матрицы переходов и расчёта вероятностей
    для системы с несколькими каналами обслуживания.
    """

    def __init__(
        self,
        params: MultiServerParams,
        config: ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    ) -> None:
        """Инициализирует многолинейную систему массового обслуживания.

        Args:
            params (MultiServerParams): Параметры системы.
            config: Конфигурация вычислений.
        """
        super().__init__(params, config)

    def _build_transition_matrix(self) -> np.ndarray[np.float64]:
        """Строит матрицу переходов многолинейной системы массового обслуживания.

        Returns:
            np.ndarray[np.float64]: Матрица переходов между состояниями системы.
        """
        transition_matrix = MultiServerMatrixBuilder(self.params).build()
        return transition_matrix


class AnalyticalMAPServerSystem(BaseAnalyticalProbabilitySystem):
    """Класс моделирования СМО с MAP-потоками и нетерпеливыми заявками.

    Реализует методы построения матрицы переходов и расчёта вероятностей
    для систем с MAP-потоками поступления заявок.
    """

    def __init__(
        self,
        params: MAPServerParams,
        config: ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    ) -> None:
        """Инициализирует систему массового обслуживания с MAP-потоками.

        Args:
            params (MAPServerParams): Параметры системы.
            config: Конфигурация вычислений.
        """
        super().__init__(params, config)

    def _build_transition_matrix(self) -> np.ndarray[np.float64]:
        """Строит матрицу переходов СМО с MAP-потоками.

        Returns:
            np.ndarray[np.float64]: Матрица переходов между состояниями системы.
        """
        transition_matrix = MAPServerMatrixBuilder(self.params).build()
        return transition_matrix
