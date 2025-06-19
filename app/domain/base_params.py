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

    def validate(self):
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
        if self.lambda_rate <= 0:
            raise ValueError("Интенсивность λ должна быть положительна.")
        if self.mu_rate <= 0:
            raise ValueError("Интенсивность μ должна быть положительна.")
        if self.max_customers <= 0:
            raise ValueError("Переменная max_customers должна быть положительна.")
        if self.time_array is None or self.time_array.shape[0] == 0:
            raise ValueError("Временной массив не задан.")
        if np.any(self.state_variables < 0):
            raise ValueError("Переменные состояния не могут быть отрицательными.")
        if self.state_variables.shape != self.initial_probabilities.shape:
            raise ValueError(
                "Размер переменных состояния должен совпадать с числом вероятностей."
            )
        if not np.isclose(np.sum(self.initial_probabilities), 1.0):
            raise ValueError("Сумма начальных вероятностей должна быть равна 1.")
        if np.any(self.initial_probabilities < 0) or np.any(
            self.initial_probabilities > 1
        ):
            raise ValueError(
                "Начальные вероятности должны находиться в диапазоне [0, 1]."
            )


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

    def validate(self):
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
        super().validate()
        if self.processor_count <= 0:
            raise ValueError("Переменная processor_count должна быть положительна.")
