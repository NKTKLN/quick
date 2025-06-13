"""Модуль для анализа пропускной способности СМО с нетерпеливыми заявками.

Этот модуль реализует классы для расчета пропускной способности
однолинейных и многолинейных систем массового обслуживания (СМО),
в которых заявки могут уходить из очереди при длительном ожидании.

Классы:
    BaseThroughputSystem: Абстрактная базовая модель для анализа СМО.
    SingleServerThroughputQueueSystem: Расчет для одноканальной СМО.
    MultiServerThroughputSystem: Расчет для многоканальной СМО.

Функциональность:
    - Серийный расчет пропускной способности при изменяющейся
      интенсивности ухода заявок (ν).
    - Использование вероятностей потери для вычисления пропускной способности:
      Throughput = (1 - P_loss) * λ.
    - Интеграция с моделями ImpatientQueueSystem и MultiImpatientQueueSystem.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
from numpy.typing import NDArray

from app.domain import (
    MultiServerParams,
    MultiServerThroughputParams,
    SingleServerParams,
    SingleServerThroughputParams,
)
from app.services.probability import MultiServerSystem, SingleServerSystem

# Настройка логирования для отслеживания работы системы
logger = logging.getLogger(__name__)


class BaseThroughputSystem(ABC):
    """Абстрактный базовый класс для анализа пропускной способности СМО.

    Определяет общий интерфейс и базовую реализацию для расчета зависимости пропускной
    способности системы от интенсивности ухода заявок. Классы-наследники реализуют
    специфичную логику для различных типов СМО (одноканальных/многоканальных).

    Основные функции:
    - Серийный расчет характеристик для диапазона значений интенсивности ухода
    - Вычисление пропускной способности как (1 - вероятность потери) * λ
    - Интеграция с базовыми моделями СМО для получения вероятностных характеристик

    Методы:
        calculate(): Основной метод для выполнения серии расчетов
        _calculate_probabilities(): Абстрактный метод расчета вероятностей
    """

    def __init__(self, params: Any) -> None:
        """Инициализирует анализатор пропускной способности с заданными параметрами.

        Args:
            params: Объект параметров, содержащий:
                   - Базовые параметры СМО (λ, μ, количество каналов и т.д.)
                   - Диапазон значений интенсивности ухода заявок (ν)
                   - Другие специфичные параметры для серийного расчета

        Note:
            Конкретный тип параметров может уточняться в классах-наследниках.
        """
        self.params = params

    def calculate(self) -> List[NDArray[np.float64]]:
        """Выполняет расчет пропускной способности для различных значений ν.

        Процесс расчета включает:
        1. Итерацию по значениям интенсивности ухода заявок (ν)
        2. Расчет вероятностных характеристик для каждого ν
        3. Вычисление пропускной способности системы
        4. Формирование итогового массива результатов

        Returns:
            List[NDArray[np.float64]]: Список массивов пропускной способности
                                       для каждого значения ν. Каждый массив
                                       соответствует результатам расчета для
                                       конкретного значения интенсивности ухода.
        """
        throughput_results = []

        # Итерация по набору параметров с разными ν
        for single_param in self.params:
            probabilities = self._calculate_probabilities(single_param)

            # Расчет пропускной способности: (1 - p_loss) * lambda
            current_throughput = (1 - probabilities[-1]) * single_param.lambda_rate
            throughput_results.append(current_throughput)

            logger.info(
                "Рассчитана пропускная способность для ν=%.3f", single_param.nu_rate
            )

        return throughput_results

    @abstractmethod
    def _calculate_probabilities(self, params: Any) -> NDArray[np.float64]:
        """Абстрактный метод для расчета вероятностей состояний системы.

        Должен быть реализован в классах-наследниках для конкретных типов СМО.

        Args:
            params: Параметры СМО для конкретного расчета

        Returns:
            NDArray[np.float64]: Массив стационарных вероятностей состояний,
                               где последний элемент соответствует вероятности
                               потери заявки.
        """
        pass


class SingleServerThroughputSystem(BaseThroughputSystem):
    """Класс для расчета пропускной способности СМО.

    Осуществляет серийный расчет характеристик СМО для различных значений
    интенсивности ухода заявок (ν) с использованием матричного метода.
    """

    def __init__(self, params: SingleServerThroughputParams) -> None:
        """Инициализирует систему с заданными параметрами.

        Args:
            params: Объект SingleServerThroughputParams, содержащий:
                    - базовые параметры СМО
                    - диапазон значений интенсивности ухода заявок (ν)
        """
        super().__init__(params)

    def _calculate_probabilities(
        self, params: SingleServerParams
    ) -> NDArray[np.float64]:
        """Вычисляет стационарные вероятности состояний СМО для заданных параметров.

        Внутренний метод, который создает экземпляр ImpatientQueueSystem с текущими
        параметрами и выполняет расчет вероятностей состояний системы.

        Args:
            params: Параметры СМО (SingleServerParams), включая:
                    - lambda_rate: интенсивность входящего потока
                    - mu_rate: интенсивность обслуживания
                    - nu_rate: интенсивность ухода заявок из очереди
                    - channel_count: количество каналов обслуживания
                    - queue_capacity: максимальная длина очереди

        Returns:
            NDArray[np.float64]: Массив стационарных вероятностей состояний СМО,
                                 где последний элемент соответствует вероятности
                                 потери заявки.
        """
        queue_system = SingleServerSystem(params)
        probabilities = queue_system.calculate()
        return probabilities


class MultiServerThroughputSystem(BaseThroughputSystem):
    """Класс для расчета пропускной способности многолинейной СМО.

    Осуществляет серийный расчет характеристик многолинейной СМО для различных значений
    интенсивности ухода заявок (ν) с использованием матричного метода.
    """

    def __init__(self, params: MultiServerThroughputParams) -> None:
        """Инициализирует систему с заданными параметрами.

        Args:
            params: Объект MultiServerThroughputParams, содержащий:
                    - базовые параметры СМО
                    - диапазон значений интенсивности ухода заявок (ν)
        """
        super().__init__(params)

    def _calculate_probabilities(
        self, params: MultiServerParams
    ) -> NDArray[np.float64]:
        """Вычисляет стационарные вероятности состояний СМО для заданных параметров.

        Внутренний метод, который создает экземпляр ImpatientQueueSystem с текущими
        параметрами и выполняет расчет вероятностей состояний системы.

        Args:
            params: Параметры СМО (MultiServerParams), включая:
                    - lambda_rate: интенсивность входящего потока
                    - mu_rate: интенсивность обслуживания
                    - nu_rate: интенсивность ухода заявок из очереди
                    - channel_count: количество каналов обслуживания
                    - queue_capacity: максимальная длина очереди
                    - processor_count (int): количество обслуживающих приборов в системе

        Returns:
            NDArray[np.float64]: Массив стационарных вероятностей состояний
                                 многолинейной СМО, где последний элемент соответствует
                                 вероятности потери заявки.
        """
        queue_system = MultiServerSystem(params)
        probabilities = queue_system.calculate()
        return probabilities
