"""Базовый модуль решателя вероятностной системы для СМО с уходом заявок.

Содержит класс `BasicProbabilitySolver`, который определяет интерфейс и
хранит общие компоненты для численного решения матричной системы уравнений
вероятностей, возникающей при моделировании одноканальных СМО с нетерпеливыми заявками.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from app.domain import ComputationConfig
from app.domain.models import SingleServerParams


class BasicProbabilitySolver(ABC):
    """Базовый класс решателя вероятностей для СМО с нетерпеливыми заявками.

    Задает общий интерфейс и базовые атрибуты для реализации численного решения
    вероятностной модели систем массового обслуживания (СМО), в которых заявки могут
    покидать очередь при длительном ожидании. Конкретные реализации решателей должны
    наследовать этот класс и переопределять соответствующие методы.
    """

    def __init__(
        self,
        params: SingleServerParams,
        coefficients_matrix: NDArray[np.float64],
        config: Optional[ComputationConfig] = None,
    ) -> None:
        """Инициализирует базовый решатель.

        Args:
            params: Параметры системы массового обслуживания.
            coefficients_matrix (NDArray[np.float64]): Матрица коэффициентов системы
                уравнений размером (n x n).
            config (Optional[ComputationConfig]): Конфигурация вычислений.
        """
        self.params = params
        self.coefficients_matrix = coefficients_matrix
        self.config = config

    @abstractmethod
    def calculate(self) -> NDArray[np.float64]:
        """Выполняет полный расчёт вероятностей состояний системы.

        Returns:
            NDArray[np.float64]: Матрица вероятностей состояний (размерность
                зависит от параметров СМО).
        """
        pass
