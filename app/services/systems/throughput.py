"""Модуль для анализа пропускной способности СМО с нетерпеливыми заявками.

Включает классы для вычисления пропускной способности одно- и многолинейных
систем массового обслуживания, в которых заявки могут покидать очередь при
длительном ожидании. Поддерживается работа с обычной и повышенной точностью
вычислений. Используется интеграция с вероятностными моделями СМО.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, cast

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
from app.services.systems.probability import (
    MAPServerSystem,
    MultiServerSystem,
    SingleServerSystem,
)

# Настройка логирования для отслеживания работы системы
logger = logging.getLogger(__name__)


class BaseThroughputSystem(ABC):
    """Базовый абстрактный класс для анализа пропускной способности СМО.

    Определяет интерфейс и общую логику вычислений зависимости пропускной способности
    от интенсивности ухода заявок из очереди. Наследники реализуют специфичные методы
    расчёта вероятностей для конкретных типов СМО.
    """

    def __init__(
        self,
        params: SingleServerParams | MultiServerParams | MAPServerParams,
        config: ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    ) -> None:
        """Инициализирует анализатор пропускной способности с заданными параметрами.

        Args:
            params: Параметры СМО с диапазоном значений интенсивности ухода заявок.
            config: Конфигурация вычислений.
        """
        self.params = params
        self.config = config

    def calculate(self) -> NDArray[np.float64] | list[NDArray[np.float64]]:
        """Выполняет полный расчёт пропускной способности системы.

        Этапы:
            1. Расчёт вероятностей состояний системы
            2. Вычисление пропускной способности: (1 - p_loss) * λ
            3. Формирование массива итоговых значений

        Returns:
            NDArray[np.float64]: Список значений пропускной способности.

        Raises:
            ValueError: Если параметры системы являются MAPServerParams.
        """
        if isinstance(self.params, MAPServerParams):
            raise ValueError("Параметры не должны быть экземпляром MAPServerParams")

        probabilities = self._calculate_probabilities(self.params)
        throughput = (1 - probabilities[-1]) * self.params.lambda_rate

        logger.info(
            "Рассчитана пропускная способность для ν = %.3f", self.params.nu_rate
        )

        return cast(NDArray, throughput)

    @abstractmethod
    def _calculate_probabilities(self, params: Any) -> NDArray[np.float64]:
        """Вычисляет вероятности состояний системы для заданных параметров.

        Args:
            params (Any): Параметры конкретного расчёта.

        Returns:
            NDArray[np.float64]: Массив вероятностей состояний,
                последний элемент — вероятность потери.
        """
        pass


class SingleServerThroughputSystem(BaseThroughputSystem):
    """Класс анализа пропускной способности однолинейной СМО.

    Реализует расчёты пропускной способности для различных значений ν с использованием
    вероятностной модели одноканальной СМО.
    """

    def __init__(
        self,
        params: SingleServerParams,
        config: ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    ) -> None:
        """Инициализирует анализатор с параметрами однолинейной системы.

        Args:
            params (SingleServerParams): Параметры СМО.
            config: Конфигурация вычислений.
        """
        super().__init__(params, config)

    def _calculate_probabilities(
        self, params: SingleServerParams
    ) -> NDArray[np.float64]:
        """Строит модель СМО и вычисляет вероятности состояний системы.

        Args:
            params (SingleServerParams): Параметры однолинейной СМО.

        Returns:
            NDArray[np.float64]: Массив вероятностей, включая вероятность потери.
        """
        queue_system = SingleServerSystem(params, self.config)
        probabilities = queue_system.calculate()
        return probabilities


class MultiServerThroughputSystem(BaseThroughputSystem):
    """Класс анализа пропускной способности многолинейной СМО.

    Выполняет серию расчётов пропускной способности при разных ν
    с использованием вероятностной модели многоканальной СМО.
    """

    def __init__(
        self,
        params: MultiServerParams,
        config: ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    ) -> None:
        """Инициализирует анализатор с параметрами многолинейной системы.

        Args:
            params (MultiServerParams): Параметры СМО.
            config: Конфигурация вычислений.
        """
        super().__init__(params, config)

    def _calculate_probabilities(
        self, params: MultiServerParams
    ) -> NDArray[np.float64]:
        """Строит модель многолинейной СМО и вычисляет вероятности состояний.

        Args:
            params (MultiServerParams): Параметры многолинейной СМО.

        Returns:
            NDArray[np.float64]: Массив вероятностей, включая вероятность потери.
        """
        queue_system = MultiServerSystem(params, self.config)
        probabilities = queue_system.calculate()
        return probabilities


class MAPServerThroughputSystem(BaseThroughputSystem):
    """Класс анализа пропускной способности СМО с MAP-потоками.

    Реализует расчёты пропускной способности для различных значений ν с использованием
    вероятностной модели СМО с MAP-потоками.
    """

    def __init__(
        self,
        params: MAPServerParams,
        config: ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    ) -> None:
        """Инициализирует анализатор с параметрами однолинейной системы.

        Args:
            params (MAPServerThroughputParams): Параметры СМО.
            config: Конфигурация вычислений.
        """
        super().__init__(params, config)

    def calculate(self) -> list[NDArray[np.float64]]:
        """Выполняет полный расчёт пропускной способности системы.

        Этапы:
            1. Итерация по значениям интенсивности поступления заявок (λ)
            2. Расчёт вероятностей состояний системы
            3. Вычисление пропускной способности: (1 - p_loss) * λ
            4. Формирование массива итоговых значений

        Returns:
            list[NDArray[np.float64]]: Список значений пропускной способности
                для каждого ν из заданного диапазона.

        Raises:
            ValueError: Если параметры системы не являются MAPServerParams.
        """
        if not isinstance(self.params, MAPServerParams):
            raise ValueError("Параметры должны быть экземпляром MAPServerParams")

        throughput_results = []

        for lambda_rate in self.params.lambda_rate:
            probabilities = self._calculate_probabilities(self.params)
            current_throughput = (1 - probabilities[-1]) * lambda_rate
            throughput_results.append(current_throughput)

            logger.info(
                "Рассчитана пропускная способность для ν = %.3f", self.params.nu_rate
            )

        return throughput_results

    def _calculate_probabilities(self, params: MAPServerParams) -> NDArray[np.float64]:
        """Строит модель СМО и вычисляет вероятности состояний системы.

        Args:
            params (MAPServerParams): Параметры СМО с MAP-потоками.

        Returns:
            NDArray[np.float64]: Массив вероятностей, включая вероятность потери.
        """
        queue_system = MAPServerSystem(params, self.config)
        probabilities = queue_system.calculate()
        return probabilities
