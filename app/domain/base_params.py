"""Базовые классы параметров систем массового обслуживания (СМО).

Содержит абстрактные классы с общими параметрами для одно- и многоканальных
систем массового обслуживания без параметра интенсивности ухода заявок (ν).
"""

from abc import ABC
from dataclasses import dataclass

import numpy as np


@dataclass
class BasicSingleServerParams(ABC):
    """Параметры одноканальной СМО без учета ухода заявок.

    Содержит базовые параметры, описывающие входной поток, обслуживание,
    ограничения по числу заявок и начальные условия системы.

    Attributes:
        lambda_rate (float): Интенсивность поступления заявок (λ > 0).
        mu_rate (float): Интенсивность обслуживания заявок (μ > 0).
        max_customers (int): Максимальное допустимое число заявок в системе (n > 0).
        time_array (np.ndarray[np.float64]): Массив временных точек расчета.
        state_variables (np.ndarray[np.float64]): Переменные состояния системы.
        initial_probabilities (np.ndarray[np.float64]): Начальные вероятности состояний.
    """

    lambda_rate: float
    mu_rate: float
    max_customers: int
    time_array: np.ndarray[np.float64]
    state_variables: np.ndarray[np.float64]
    initial_probabilities: np.ndarray[np.float64]


@dataclass
class BasicMultiServerParams(BasicSingleServerParams):
    """Параметры многоканальной СМО без учета ухода заявок.

    Расширяет параметры одноканальной системы, добавляя количество
    обслуживающих каналов (процессоров).

    Attributes:
        processor_count (int): Количество обслуживающих каналов (m).
        Остальные параметры наследуются от BasicSingleServerParams.
    """

    processor_count: int
