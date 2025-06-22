"""Модуль для моделирования СМО с нетерпеливыми заявками.

Включает классы для вычисления вероятностных характеристик одно- и многолинейных СМО,
в том числе с MAP-потоками поступления. Используются матричные методы.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from app.domain import (
    ComputationConfig,
    MergedComputationConfig,
    MpmathComputationConfig,
)
from app.domain.models import (
    MAPServerParams,
    MultiServerParams,
    SingleServerParams,
)
from app.services import (
    MAPServerMatrixBuilder,
    MultiServerMatrixBuilder,
    SingleServerMatrixBuilder,
)
from app.services.solvers import solvers_factory


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
            params: Параметры СМО (интенсивности, структура, др.).
            config: Конфигурация вычислений.
        """
        self.params = params
        self.config = config

    def calculate(self) -> NDArray[np.float64]:
        """Выполняет полный расчёт вероятностей состояний системы.

        Returns:
            NDArray[np.float64]: Матрица вероятностей состояний (размерность
                зависит от параметров СМО).
        """
        transition_matrix = self._build_transition_matrix()
        prob_solver = solvers_factory(
            calculation_method=self.config.calculation_method,
            calculation_engine=self.config.calculation_engine,
            params=self.params,
            coefficients_matrix=transition_matrix,
            config=self.config,
        )
        return prob_solver.calculate()

    @abstractmethod
    def _build_transition_matrix(self) -> NDArray[np.float64]:
        """Строит матрицу переходов между состояниями СМО.

        Returns:
            NDArray[np.float64]: Матрица переходов.
        """
        pass


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

    def _build_transition_matrix(self) -> NDArray[np.float64]:
        """Строит матрицу переходов однолинейной системы.

        Returns:
            NDArray[np.float64]: Матрица переходов между состояниями системы.

        Raises:
            ValueError: Если параметры системы не являются SingleServerParams.
        """
        if not isinstance(self.params, SingleServerParams):
            raise ValueError("Параметры должны быть экземпляром SingleServerParams")

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

    def _build_transition_matrix(self) -> NDArray[np.float64]:
        """Строит матрицу переходов многолинейной системы массового обслуживания.

        Returns:
            NDArray[np.float64]: Матрица переходов между состояниями системы.

        Raises:
            ValueError: Если параметры системы не являются MultiServerParams.
        """
        if not isinstance(self.params, MultiServerParams):
            raise ValueError("Параметры должны быть экземпляром MultiServerParams")

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

    def _build_transition_matrix(self) -> NDArray[np.float64]:
        """Строит матрицу переходов СМО с MAP-потоками.

        Returns:
            NDArray[np.float64]: Матрица переходов между состояниями системы.

        Raises:
            ValueError: Если параметры системы не являются MAPServerParams.
        """
        if not isinstance(self.params, MAPServerParams):
            raise ValueError("Параметры должны быть экземпляром MAPServerParams")

        transition_matrix = MAPServerMatrixBuilder(self.params).build()
        return transition_matrix
