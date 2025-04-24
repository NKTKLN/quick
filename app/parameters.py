"""Модуль с классами параметров для систем массового обслуживания.

Содержит набор dataclass-ов для хранения и обработки параметров:
- Одноканальных СМО с нетерпеливыми заявками
- Многоканальных СМО
- СМО с анализом пропускной способности при различных параметрах

Основные классы:
1. QueueSystemParameters - базовые параметры одноканальной СМО
2. ThroughputQueueSystemParameters - параметры для анализа пропускной способности
3. MultiQueueSystemParameters - параметры многоканальной СМО
"""

from abc import ABC
from dataclasses import dataclass, fields
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


class BaseQueueSystemParameters(ABC):
    """Базовый класс для всех систем массового обслуживания.

    Содержит общие атрибуты для систем массового обслуживания без параметра nu_rate.

    Attributes:
        lambda_rate (float): Интенсивность входящего потока заявок (λ > 0)
        mu_rate (float): Интенсивность обслуживания заявок (μ > 0)
        max_customers (int): Максимальная емкость системы - число заявок (n > 0)
        time_array (NDArray[np.float64]): Временная сетка для расчета (t_i),
                                          упорядоченный массив временных точек
        state_variables (NDArray[np.float64]): Вектор переменных состояния системы
        initial_probabilities (NDArray[np.float64]): Начальное распределение
                                                     вероятностей состояний системы
    """

    lambda_rate: float  # Интенсивность поступления заявок (λ)
    mu_rate: float  # Интенсивность обслуживания заявок (μ)
    max_customers: int  # Максимальное количество заявок в системе (n)
    time_array: NDArray[np.float64]  # Массив времени
    state_variables: NDArray[np.float64]  # Массив переменных состояния
    initial_probabilities: NDArray[np.float64]  # Массив начальных вероятностей


class MultiBaseQueueSystemParameters(BaseQueueSystemParameters):
    """Расширение параметров для многолинейной системы массового обслуживания.

    Добавляет параметр количества обслуживающих приборов (процессоров) к базовым
    параметрам СМО.

    Attributes:
        processor_count (int): Количество обслуживающих приборов (процессоров) в системе
        Все остальные атрибуты наследуются от BaseQueueSystemParameters
    """

    processor_count: int  # Количество обслуживающих процессоров (m)


@dataclass
class QueueSystemParameters(BaseQueueSystemParameters):
    """Параметры системы массового обслуживания с нетерпеливыми заявками.

    Содержит все необходимые параметры для моделирования СМО с нетерпеливыми заявками
    в переходном режиме. Все параметры обязательны для корректной работы модели.

    Attributes:
        nu_rate (float): Интенсивность ухода нетерпеливых заявок из очереди (ν ≥ 0)
        Все остальные атрибуты наследуются от BaseQueueSystemParameters
    """

    nu_rate: float  # Интенсивность ухода нетерпеливых заявок (ν)


@dataclass
class ThroughputQueueSystemParameters(BaseQueueSystemParameters):
    """Расширение параметров СМО для анализа пропускной способности при различных ν.

    Позволяет задать массив значений интенсивности ухода заявок (ν) и итерироваться
    по ним, возвращая на каждой итерации экземпляр QueueSystemParameters.

    Attributes:
        nu_rate (NDArray[np.float64]): Массив интенсивностей ухода
                                       нетерпеливых заявок (ν)
        Все остальные атрибуты наследуются от BaseQueueSystemParameters
    """

    nu_rate: NDArray[np.float64]  # Список интенсивностей ухода нетерпеливых заявок (ν)

    def __iter__(self) -> Iterator[QueueSystemParameters]:
        """Итератор по всем значениям ν, возвращающий QueueSystemParameters.

        Returns:
            Iterator[QueueSystemParameters]: Итератор, который для каждого значения ν
            из массива nu_rate возвращает полный набор параметров СМО

        Note:
            При итерации создаются новые экземпляры QueueSystemParameters с
            фиксированным значением ν из массива nu_rate, все остальные параметры
            копируются из текущего объекта.
        """
        # Сборка параметров для базового QueueSystemParameters без nu_rate
        base_params = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "nu_rate"
        }

        # Генерация экземпляров QueueSystemParameters для каждого значения nu_rate
        for nu in self.nu_rate:
            yield QueueSystemParameters(**base_params, nu_rate=nu)


@dataclass
class MultiQueueSystemParameters(MultiBaseQueueSystemParameters):
    """Параметры многолинейной системы массового обслуживания с нетерпеливыми заявками.

    Содержит все необходимые параметры для моделирования многолинейной СМО с
    нетерпеливыми заявками в переходном режиме. Все параметры обязательны для
    корректной работы модели.

    Attributes:
        nu_rate (float): Интенсивность ухода нетерпеливых заявок из очереди (ν ≥ 0)
        Все остальные атрибуты наследуются от MultiBaseQueueSystemParameters
    """

    nu_rate: float  # Интенсивность ухода нетерпеливых заявок (ν)


@dataclass
class MultiThroughputQueueSystemParameters(MultiBaseQueueSystemParameters):
    """Расширение параметров многолинейной СМО для анализа пропускной способности.

    Позволяет задать массив значений интенсивности ухода заявок (ν) и итерироваться
    по ним, возвращая на каждой итерации экземпляр MultiQueueSystemParameters.

    Attributes:
        nu_rate (NDArray[np.float64]): Массив интенсивностей ухода
                                       нетерпеливых заявок (ν)
        Все остальные атрибуты наследуются от MultiBaseQueueSystemParameters
    """

    nu_rate: NDArray[np.float64]  # Список интенсивностей ухода нетерпеливых заявок (ν)

    def __iter__(self) -> Iterator[MultiQueueSystemParameters]:
        """Итератор по всем значениям ν, возвращающий MultiQueueSystemParameters.

        Returns:
            Iterator[MultiQueueSystemParameters]: Итератор, который для каждого
            значения ν из массива nu_rate возвращает полный набор параметров СМО

        Note:
            При итерации создаются новые экземпляры MultiQueueSystemParameters с
            фиксированным значением ν из массива nu_rate, все остальные параметры
            копируются из текущего объекта.
        """
        # Сборка параметров для базового MultiQueueSystemParameters без nu_rate
        base_params = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "nu_rate"
        }

        # Генерация экземпляров MultiQueueSystemParameters для каждого значения nu_rate
        for nu in self.nu_rate:
            yield MultiQueueSystemParameters(**base_params, nu_rate=nu)
