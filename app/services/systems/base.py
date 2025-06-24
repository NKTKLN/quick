"""Модуль базовой модели системы массового обслуживания (СМО).

Содержит абстрактный класс `BaseServerSystem`, определяющий интерфейс для вычисления
основных характеристик СМО с нетерпеливыми заявками.
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


class BaseServerSystem(ABC):
    """Абстрактный базовый класс для систем массового обслуживания с уходами заявок.

    Задает интерфейс и общую структуру для всех моделей СМО, включая параметры
    системы и конфигурацию вычислений. Является основой для всех специализированных
    реализаций СМО (одноканальные, многоканальные, MAP-потоки и др.) и их расчётов.
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

    @abstractmethod
    def calculate(self) -> NDArray[np.float64]:
        """Выполняет полный расчёт значений состояний системы.

        Returns:
            NDArray[np.float64]: Матрица значений
                состояний (размерность зависит от параметров СМО).
        """
        raise NotImplementedError()
