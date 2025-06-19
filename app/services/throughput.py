"""Модуль для анализа пропускной способности СМО с нетерпеливыми заявками.

Включает классы для вычисления пропускной способности одно- и многолинейных
систем массового обслуживания, в которых заявки могут покидать очередь при
длительном ожидании. Поддерживается работа с обычной и повышенной точностью
вычислений. Используется интеграция с вероятностными моделями СМО.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from app.domain import (
    ComputationConfig,
    MergedComputationConfig,
    MpmathComputationConfig,
    MultiServerParams,
    MultiServerThroughputParams,
    SingleServerParams,
    SingleServerThroughputParams,
)
from app.services.probability import MultiServerSystem, SingleServerSystem

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
        params: SingleServerThroughputParams | MultiServerThroughputParams,
        config: ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    ) -> None:
        """Инициализирует анализатор пропускной способности с заданными параметрами.

        Args:
            params (SingleServerThroughputParams | MultiServerThroughputParams):
                Параметры СМО с диапазоном значений интенсивности ухода заявок.
            config: Конфигурация вычислений.
        """
        self.params = params
        self.config = config

    def calculate(self) -> list[np.ndarray[np.float64]]:
        """Выполняет полный расчёт пропускной способности системы.

        Этапы:
            1. Итерация по значениям интенсивности ухода заявок (ν)
            2. Расчёт вероятностей состояний системы
            3. Вычисление пропускной способности: (1 - p_loss) * λ
            4. Формирование массива итоговых значений

        Returns:
            list[np.ndarray[np.float64]]: Список значений пропускной способности
                для каждого ν из заданного диапазона.
        """
        throughput_results = []

        for single_param in self.params:
            probabilities = self._calculate_probabilities(single_param)
            current_throughput = (1 - probabilities[-1]) * single_param.lambda_rate
            throughput_results.append(current_throughput)

            logger.info(
                "Рассчитана пропускная способность для ν = %.3f", single_param.nu_rate
            )

        return throughput_results

    @abstractmethod
    def _calculate_probabilities(self, params: Any) -> np.ndarray[np.float64]:
        """Вычисляет вероятности состояний системы для заданных параметров.

        Args:
            params (Any): Параметры конкретного расчёта.

        Returns:
            np.ndarray[np.float64]: Массив вероятностей состояний,
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
        queue_system = MultiServerSystem(params, self.config)
        probabilities = queue_system.calculate()
        return probabilities
