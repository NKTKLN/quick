"""Модуль для моделирования систем массового обслуживания с нетерпеливыми заявками.

Включает классы для вычисления вероятностных характеристик одно- и многолинейных СМО,
в том числе с MAP-потоками поступления. Используются матричные методы. Поддерживаются
вычисления с обычной точностью (numpy) и повышенной точностью (mpmath).
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import mpmath as mp  # type: ignore[import-untyped]
import numpy as np
from scipy.linalg import eig as scipy_eig

from app.domain import (
    CalculationType,
    ComputationConfig,
    MAPServerParams,
    MergedComputationConfig,
    MpmathComputationConfig,
    MultiServerParams,
    SingleServerParams,
)
from app.services.matrix_generators import (
    MAPServerMatrixBuilder,
    MultiServerMatrixBuilder,
    SingleServerMatrixBuilder,
)
from app.services.solver import (
    MergedProbabilitySolver,
    MpmathProbabilitySolver,
    NumpyProbabilitySolver,
)

# Настройка логирования для отслеживания работы системы
logger = logging.getLogger(__name__)


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
        if self.config.calculation_type in [
            CalculationType.MPMATH,
            CalculationType.MERGED,
        ] and isinstance(
            self.config, MpmathComputationConfig | MergedComputationConfig
        ):
            with mp.workdps(self.config.precision):
                mp_matrix = mp.matrix(transition_matrix.tolist())
                eigenvalues, xsi_matrix = mp.eig(mp_matrix)
            return eigenvalues, xsi_matrix

        eigenvalues, xsi_matrix = scipy_eig(transition_matrix)
        return eigenvalues, xsi_matrix

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
        match self.config.calculation_type:
            case CalculationType.NUMPY if isinstance(self.config, ComputationConfig):
                prob_solver = NumpyProbabilitySolver(
                    self.params, transition_matrix, eigenvalues, self.config
                )
            case CalculationType.MPMATH if isinstance(
                self.config, MpmathComputationConfig
            ):
                prob_solver = MpmathProbabilitySolver(
                    self.params, transition_matrix, eigenvalues, self.config
                )
            case CalculationType.MERGED if isinstance(
                self.config, MergedComputationConfig
            ):
                prob_solver = MergedProbabilitySolver(
                    self.params, transition_matrix, eigenvalues, self.config
                )
            case _:
                raise ValueError("Unsupported calculation type")

        m_matrix = prob_solver.generate_m_matrix(xsi_matrix)
        return prob_solver.generate_p_matrix(m_matrix)


class SingleServerSystem(BaseProbabilitySystem):
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


class MultiServerSystem(BaseProbabilitySystem):
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


class MAPServerSystem(BaseProbabilitySystem):
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
