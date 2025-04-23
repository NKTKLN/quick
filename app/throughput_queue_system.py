"""Модуль для анализа пропускной способности СМО с нетерпеливыми заявками.

Модуль содержит классы для расчета пропускной способности однолинейных и
многолинейных СМО при различных значениях интенсивности ухода заявок.

Основные классы:
1. ThroughputQueueSystem - расчет пропускной способности однолинейной СМО
2. MultiThroughputQueueSystem - расчет пропускной способности многолинейной СМО

Основные функции:
- Серийный расчет характеристик для диапазона значений интенсивности ухода
- Вычисление пропускной способности как (1 - вероятность потери) * интенсивность входа
- Интеграция с базовыми моделями СМО для получения вероятностных характеристик
"""

import logging
from typing import List

import numpy as np
from numpy.typing import NDArray

from app.impatient_queue_system import ImpatientQueueSystem, MultiImpatientQueueSystem
from app.parameters import (
    MultiQueueSystemParameters,
    MultiThroughputQueueSystemParameters,
    QueueSystemParameters,
    ThroughputQueueSystemParameters,
)

# Настройка логирования для отслеживания работы системы
logger = logging.getLogger(__name__)


class ThroughputQueueSystem:
    """Класс для расчета пропускной способности СМО.

    Осуществляет серийный расчет характеристик СМО для различных значений
    интенсивности ухода заявок (ν) с использованием матричного метода.
    """

    def __init__(self, params: ThroughputQueueSystemParameters) -> None:
        """Инициализирует систему с заданными параметрами.

        Args:
            params: Объект ThroughputQueueSystemParameters, содержащий:
                    - базовые параметры СМО
                    - диапазон значений интенсивности ухода заявок (ν)
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

    def _calculate_probabilities(
        self, params: QueueSystemParameters
    ) -> NDArray[np.float64]:
        """Вычисляет стационарные вероятности состояний СМО для заданных параметров.

        Внутренний метод, который создает экземпляр ImpatientQueueSystem с текущими
        параметрами и выполняет расчет вероятностей состояний системы.

        Args:
            params: Параметры СМО (QueueSystemParameters), включая:
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
        queue_system = ImpatientQueueSystem(params)
        probabilities = queue_system.calculate()
        return probabilities


class MultiThroughputQueueSystem:
    """Класс для расчета пропускной способности многолинейной СМО.

    Осуществляет серийный расчет характеристик многолинейной СМО для различных значений
    интенсивности ухода заявок (ν) с использованием матричного метода.
    """

    def __init__(self, params: MultiThroughputQueueSystemParameters) -> None:
        """Инициализирует систему с заданными параметрами.

        Args:
            params: Объект MultiThroughputQueueSystemParameters, содержащий:
                    - базовые параметры СМО
                    - диапазон значений интенсивности ухода заявок (ν)
        """
        self.params = params

    def _calculate_probabilities(
        self, params: MultiQueueSystemParameters
    ) -> NDArray[np.float64]:
        """Вычисляет стационарные вероятности состояний СМО для заданных параметров.

        Внутренний метод, который создает экземпляр ImpatientQueueSystem с текущими
        параметрами и выполняет расчет вероятностей состояний системы.

        Args:
            params: Параметры СМО (MultiQueueSystemParameters), включая:
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
        queue_system = MultiImpatientQueueSystem(params)
        probabilities = queue_system.calculate()
        return probabilities
