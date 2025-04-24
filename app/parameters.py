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
from typing import Any, Iterator, Type

import numpy as np
from numpy.typing import NDArray


class QueueSystemParametersIterator:
    """Итератор для генерации параметров СМО с разными значениями интенсивности ухода.

    Позволяет последовательно получать параметры СМО для каждого значения
    интенсивности ухода заявок (ν) из заданного массива.

    Attributes:
        nu_rate (NDArray[np.float64]): Массив значений интенсивности ухода заявок
        base_params (dict[str, Any]): Базовые параметры СМО (без ν)
        class_type (Type[Union[QueueSystemParameters, MultiQueueSystemParameters]]):
            Тип класса параметров для создания экземпляров
        __position (int): Текущая позиция в массиве nu_rate
    """

    def __init__(
        self,
        nu_rate: NDArray[np.float64],
        base_params: dict[str, Any],
        class_type: Type,
    ) -> None:
        """Инициализация итератора.

        Args:
            nu_rate: Массив значений интенсивности ухода заявок (ν)
            base_params: Базовые параметры СМО (все, кроме ν)
            class_type: Класс параметров СМО для создания экземпляров
        """
        self.nu_rate = nu_rate
        self.base_params = base_params
        self.class_type = class_type
        self.__position = 0

    def __iter__(
        self,
    ) -> Iterator:
        """Возвращает сам итератор.

        Returns:
            Iterator: Сам объект итератора
        """
        return self

    def __next__(self) -> Any:
        """Возвращает следующий набор параметров СМО.

        Returns:
            QueueSystemParameters | MultiQueueSystemParameters:
            Экземпляр класса параметров СМО с очередным значением ν из массива

        Raises:
            StopIteration: Когда достигнут конец массива nu_rate
        """
        if self.__position < len(self.nu_rate):
            nu = self.nu_rate[self.__position]
            self.__position += 1
            return self.class_type(**self.base_params, nu_rate=nu)
        raise StopIteration


@dataclass
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


@dataclass
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
            из массива nu_rate возвращает полный набор параметров СМО.

        Note:
            При итерации создаются новые экземпляры QueueSystemParameters с
            фиксированным значением ν из массива nu_rate, все остальные параметры
            копируются из текущего объекта.
        """
        base_params = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "nu_rate"
        }
        return QueueSystemParametersIterator(
            self.nu_rate, base_params, QueueSystemParameters
        )


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

    def __iter__(self) -> QueueSystemParametersIterator:
        """Итератор по всем значениям ν, возвращающий MultiQueueSystemParameters.

        Returns:
            MultiQueueSystemParameters: Итератор, который для каждого
            значения ν из массива nu_rate возвращает полный набор параметров СМО

        Note:
            При итерации создаются новые экземпляры MultiQueueSystemParameters с
            фиксированным значением ν из массива nu_rate, все остальные параметры
            копируются из текущего объекта.
        """
        base_params = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "nu_rate"
        }
        return QueueSystemParametersIterator(
            self.nu_rate, base_params, MultiQueueSystemParameters
        )
