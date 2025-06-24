"""Модуль для моделирования СМО с нетерпеливыми заявками.

Включает класс для вычисления вероятностных характеристик одно- и многолинейных СМО,
в том числе с MAP-потоками поступления. Используются матричные методы.
"""

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
from app.services.matrix_builders import matrix_builder_factory
from app.services.solvers import solvers_factory
from app.services.systems.base import BaseSystem


class ServerProbabilitySystem(BaseSystem):
    """Класс для моделирования СМО с нетерпеливыми заявками.

    Реализует методы построения матрицы переходов и расчёта вероятностей состояний СМО.
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
        super().__init__(params, config)

    def calculate(self) -> NDArray[np.float64]:
        """Выполняет полный расчёт вероятностей состояний системы.

        Returns:
            NDArray[np.float64]: Матрица вероятностей состояний (размерность
                зависит от параметров СМО).
        """
        transition_matrix = matrix_builder_factory(self.params).build()
        prob_solver = solvers_factory(
            calculation_method=self.config.calculation_method,
            calculation_engine=self.config.calculation_engine,
            params=self.params,
            coefficients_matrix=transition_matrix,
            config=self.config,
        )
        return prob_solver.calculate()
