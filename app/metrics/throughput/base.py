"""Модуль базового абстрактного класса для анализа пропускной способности СМО.

Содержит класс BaseThroughputSystem, который задаёт общий интерфейс и логику
вычисления пропускной способности с учётом интенсивности ухода заявок из очереди.
Наследники реализуют конкретные методы вычисления вероятностей для различных типов СМО.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from app.domain import (
    ComputationConfig,
    MergedComputationConfig,
    MpmathComputationConfig,
)
from app.domain.models import (
    MAPServerThroughputParams,
    MultiServerThroughputParams,
    SingleServerThroughputParams,
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
        params: (
            SingleServerThroughputParams
            | MultiServerThroughputParams
            | MAPServerThroughputParams
        ),
        config: ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    ) -> None:
        """Инициализирует анализатор пропускной способности с заданными параметрами.

        Args:
            params: Параметры СМО с диапазоном значений интенсивности ухода заявок.
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
