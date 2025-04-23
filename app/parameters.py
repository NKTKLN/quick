"""Модуль с параметрами системы массового обслуживания (СМО).

Содержит dataclass QueueSystemParameters, инкапсулирующий параметры
для моделирования СМО с нетерпеливыми заявками.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class QueueSystemParameters:
    """Параметры системы массового обслуживания с нетерпеливыми заявками.

    Содержит все необходимые параметры для моделирования СМО с нетерпеливыми заявками
    в переходном режиме. Все параметры обязательны для корректной работы модели.

    Attributes:
        lambda_rate (float): Интенсивность входящего потока заявок (λ > 0)
        mu_rate (float): Интенсивность обслуживания заявок (μ > 0)
        nu_rate (float): Интенсивность ухода нетерпеливых заявок из очереди (ν ≥ 0)
        max_customers (int): Максимальная емкость системы - число заявок (n > 0)
        time_array (NDArray[np.float64]): Временная сетка для расчета (t_i),
                                          упорядоченный массив временных точек
        state_variables (NDArray[np.float64]): Вектор переменных состояния системы
        initial_probabilities (NDArray[np.float64]): Начальное распределение
                                                     вероятностей состояний системы

    Example:
        >>> params = QueueSystemParameters(
        ...     lambda_rate=1.5,
        ...     mu_rate=1.0,
        ...     nu_rate=0.5,
        ...     max_customers=10,
        ...     time_array=np.linspace(0, 10, 100),
        ...     state_variables=np.zeros(10),
        ...     initial_probabilities=np.array([1.0] + [0.0]*9)
        ... )
    """

    lambda_rate: float  # Интенсивность поступления заявок (λ)
    mu_rate: float  # Интенсивность обслуживания заявок (μ)
    nu_rate: float  # Интенсивность ухода нетерпеливых заявок (ν)
    max_customers: int  # Максимальное количество заявок в системе (n)
    time_array: NDArray[np.float64]  # Массив времени
    state_variables: NDArray[np.float64]  # Массив переменных состояния
    initial_probabilities: NDArray[np.float64]  # Массив начальных вероятностей
