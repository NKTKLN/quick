"""Базовый абстрактный класс для расчёта вероятностей в системах массового обслуживания.

Модуль содержит класс BaseProbabilitySystem, обеспечивающий интерфейс и методы
для построения матриц переходов, вычисления собственных значений и решения
системы уравнений переходов для стационарных вероятностей.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from app.domain.models import MAPServerParams, MultiServerParams, SingleServerParams


class BaseProbabilitySystem(ABC):
    """Базовый абстрактный класс для моделирования СМО с нетерпеливыми заявками.

    Определяет интерфейс и общую логику расчёта вероятностей состояний СМО.
    Наследники реализуют специфичные методы построения матриц переходов.
    """

    def __init__(
        self,
        params: SingleServerParams | MultiServerParams | MAPServerParams,
    ) -> None:
        """Инициализирует систему массового обслуживания с заданными параметрами.

        Args:
            params (SingleServerParams | MultiServerParams | MAPServerParams):
                Параметры СМО (интенсивности, структура, др.).
        """
        self.params = params

    @abstractmethod
    def calculate(self) -> NDArray[np.float64]:
        """Выполняет полный расчёт вероятностей состояний системы.

        Returns:
            NDArray[np.float64]: Матрица вероятностей состояний (размерность
                зависит от параметров СМО).
        """
        pass

    @abstractmethod
    def _build_transition_matrix(self) -> NDArray[np.float64]:
        """Строит матрицу переходов между состояниями СМО.

        Returns:
            NDArray[np.float64]: Матрица переходов.
        """
        pass
