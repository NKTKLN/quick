"""Базовые классы параметров систем массового обслуживания (СМО).

Содержит абстрактные классы с общими параметрами для одно- и многолинейных
систем массового обслуживания без параметра интенсивности ухода заявок (ν).
"""

from abc import ABC
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class BasicSingleServerParams(ABC):
    """Базовый класс параметров системы массового обслуживания.

    Содержит основные атрибуты, необходимые для описания однолинейной системы
    массового обслуживания без учета ухода нетерпеливых заявок.

    Атрибуты:
        lambda_rate (float): Интенсивность входящего потока заявок (λ > 0).
        mu_rate (float): Интенсивность обслуживания заявок (μ > 0).
        max_customers (int): Максимальное число заявок (n > 0).
        time_array (NDArray[np.float64]): Упорядоченный массив временных точек.
        state_variables (NDArray[np.float64]): Вектор переменных состояния системы.
        initial_probabilities (NDArray[np.float64]): Начальное распределение
                                                     вероятностей состояний.
    """

    lambda_rate: float  # Интенсивность поступления заявок (λ)
    mu_rate: float  # Интенсивность обслуживания заявок (μ)
    max_customers: int  # Максимальное количество заявок в системе (n)
    time_array: NDArray[np.float64]  # Массив времени
    state_variables: NDArray[np.float64]  # Массив переменных состояния
    initial_probabilities: NDArray[np.float64]  # Массив начальных вероятностей


@dataclass
class BasicMultiServerParams(BasicSingleServerParams):
    """Базовый класс параметров многолинейной системы массового обслуживания.

    Расширяет базовый класс параметров СМО, добавляя количество обслуживающих
    приборов (процессоров) в системе.

    Атрибуты:
        processor_count (int): Количество обслуживающих приборов в системе.
        Остальные атрибуты наследуются от BasicSingleServerParams.
    """

    processor_count: int  # Количество обслуживающих процессоров (m)
