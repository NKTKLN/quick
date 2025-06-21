"""Модуль для анализа пропускной способности СМО с нетерпеливыми заявками.

Включает классы для вычисления пропускной способности одно- и многолинейных
систем массового обслуживания, в которых заявки могут покидать очередь при
длительном ожидании. Поддерживается работа с обычной и повышенной точностью
вычислений. Используется интеграция с вероятностными моделями СМО.
"""

import logging

import numpy as np

from app.domain import (
    ComputationConfig,
    MergedComputationConfig,
    MpmathComputationConfig,
)
from app.domain.models import (
    MAPServerParams,
    MAPServerThroughputParams,
    MultiServerParams,
    MultiServerThroughputParams,
    SingleServerParams,
    SingleServerThroughputParams,
)
from app.metrics.probability import (
    AnalyticalMAPServerSystem,
    AnalyticalMultiServerSystem,
    AnalyticalSingleServerSystem,
)
from app.metrics.throughput.base import BaseThroughputSystem

# Настройка логирования для отслеживания работы системы
logger = logging.getLogger(__name__)


class AnalyticalSingleServerThroughputSystem(BaseThroughputSystem):
    """Класс анализа пропускной способности однолинейной СМО.

    Реализует расчёты пропускной способности для различных значений ν с использованием
    вероятностной модели одноканальной СМО.
    """

    def __init__(
        self,
        params: SingleServerThroughputParams,
        config: ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    ) -> None:
        """Инициализирует анализатор с параметрами однолинейной системы.

        Args:
            params (SingleServerThroughputParams): Параметры СМО.
            config: Конфигурация вычислений.
        """
        super().__init__(params, config)

    def _calculate_probabilities(
        self, params: SingleServerParams
    ) -> np.ndarray[np.float64]:
        """Строит модель СМО и вычисляет вероятности состояний системы.

        Args:
            params (SingleServerParams): Параметры однолинейной СМО.

        Returns:
            np.ndarray[np.float64]: Массив вероятностей, включая вероятность потери.
        """
        queue_system = AnalyticalSingleServerSystem(params, self.config)
        probabilities = queue_system.calculate()
        return probabilities


class AnalyticalMultiServerThroughputSystem(BaseThroughputSystem):
    """Класс анализа пропускной способности многолинейной СМО.

    Выполняет серию расчётов пропускной способности при разных ν
    с использованием вероятностной модели многоканальной СМО.
    """

    def __init__(
        self,
        params: MultiServerThroughputParams,
        config: ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    ) -> None:
        """Инициализирует анализатор с параметрами многолинейной системы.

        Args:
            params (MultiServerThroughputParams): Параметры СМО.
            config: Конфигурация вычислений.
        """
        super().__init__(params, config)

    def _calculate_probabilities(
        self, params: MultiServerParams
    ) -> np.ndarray[np.float64]:
        """Строит модель многолинейной СМО и вычисляет вероятности состояний.

        Args:
            params (MultiServerParams): Параметры многолинейной СМО.

        Returns:
            np.ndarray[np.float64]: Массив вероятностей, включая вероятность потери.
        """
        queue_system = AnalyticalMultiServerSystem(params, self.config)
        probabilities = queue_system.calculate()
        return probabilities


class AnalyticalMAPServerThroughputSystem(BaseThroughputSystem):
    """Класс анализа пропускной способности СМО с MAP-потоками.

    Реализует расчёты пропускной способности для различных значений ν с использованием
    вероятностной модели СМО с MAP-потоками.
    """

    def __init__(
        self,
        params: MAPServerThroughputParams,
        config: ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    ) -> None:
        """Инициализирует анализатор с параметрами однолинейной системы.

        Args:
            params (MAPServerThroughputParams): Параметры СМО.
            config: Конфигурация вычислений.
        """
        super().__init__(params, config)

    def _calculate_probabilities(
        self, params: MAPServerParams
    ) -> np.ndarray[np.float64]:
        """Строит модель СМО и вычисляет вероятности состояний системы.

        Args:
            params (MAPServerParams): Параметры СМО с MAP-потоками.

        Returns:
            np.ndarray[np.float64]: Массив вероятностей, включая вероятность потери.
        """
        queue_system = AnalyticalMAPServerSystem(params, self.config)
        probabilities = queue_system.calculate()
        return probabilities
