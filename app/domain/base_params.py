"""Базовые классы параметров систем массового обслуживания (СМО).

Содержит абстрактные классы с общими параметрами для одно- и многоканальных
систем массового обслуживания без параметра интенсивности ухода заявок (ν).
"""

from abc import ABC
from dataclasses import dataclass

import numpy as np


@dataclass
class BasicServerParams(ABC):
    """Параметры СМО без учета поступления и ухода заявок.

    Содержит базовые параметры, описывающие входной поток, обслуживание,
    ограничения по числу заявок и начальные условия системы.

    Attributes:
        mu_rate (float): Интенсивность обслуживания заявок (μ > 0).
        max_customers (int): Максимальное допустимое число заявок в системе (n > 0).
        time_array (np.ndarray[np.float64]): Массив временных точек расчета.
        state_variables (np.ndarray[np.float64]): Переменные состояния системы.
        initial_probabilities (np.ndarray[np.float64]): Начальные вероятности состояний.
    """

    mu_rate: float
    max_customers: int
    time_array: np.ndarray[np.float64]
    state_variables: np.ndarray[np.float64]
    initial_probabilities: np.ndarray[np.float64]

    def validate(self):
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
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
class BasicSingleServerParams(BasicServerParams):
    """Параметры одноканальной СМО без учета ухода заявок.

    Содержит базовые параметры, описывающие входной поток, обслуживание,
    ограничения по числу заявок и начальные условия системы.

    Attributes:
        lambda_rate (float): Интенсивность поступления заявок (λ > 0).
        Остальные параметры наследуются от BasicServerParams.
    """

    lambda_rate: float

    def validate(self):
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
        super().validate()
        if self.lambda_rate <= 0:
            raise ValueError("Интенсивность λ должна быть положительна.")
        if self.initial_probabilities.shape[0] != self.max_customers:
            raise ValueError(
                "Размер начальных вероятностей должен совпадать с максимальным числом "
                "заявок в системе."
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
        if (
            self.initial_probabilities.shape[0]
            != self.max_customers + self.processor_count + 1
        ):
            raise ValueError(
                "Размер начальных вероятностей должен совпадать с максимальным числом "
                "заявок в системе + колличество процессоров + 1."
            )


@dataclass
class BasicMAPServerParams(BasicServerParams):
    """Параметры СМО с MAP-потоками и уходом нетерпеливых заявок.

    Описывает систему с марковским модулированным пуассоновским входным потоком,
    а также оттоком нетерпеливых заявок. Включает матрицы интенсивностей переходов.

    Attributes:
        lambda_rate (np.ndarray[np.float64]): Массив значений интенсивности
            поступления заявок (λ > 0).
        p_rate (np.ndarray[np.float64]): Матрица интенсивностей обслуживания.
        q_rate (np.ndarray[np.float64]): Матрица интенсивностей поступления.
        Остальные параметры наследуются от BasicSingleServerParams.
    """  # TODO: Проверить корректность описания p_rate и q_rate

    lambda_rate: np.ndarray[np.float64]
    p_rate: np.ndarray[np.float64]
    q_rate: np.ndarray[np.float64]

    def validate(self):
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
        super().validate()
        if np.any(self.lambda_rate <= 0) or self.lambda_rate.shape[0] == 0:
            raise ValueError("Интенсивность λ должна быть положительна.")
        if self.initial_probabilities.shape[0] != self.max_customers**2:
            raise ValueError(
                "Размер начальных вероятностей должен совпадать с максимальным числом "
                "заявок в системе возведенных в квадрат."
            )
        if self.p_rate.shape != self.q_rate.shape:
            raise ValueError(
                "Размер начальных вероятностей должен совпадать с максимальным числом "
                "заявок в системе."
            )
        if self.p_rate.ndim != 2 or self.p_rate.shape[0] != self.p_rate.shape[1]:
            raise ValueError(
                "Матрицы интенсивностей должны быть размерности 2D и быть квадратными."
            )
        if np.any((self.p_rate < 0) | (self.p_rate > 1)):
            raise ValueError(
                "Значения матрицы интенсивности обслуживания должны находиться в "
                "диапазоне [0, 1]."
            )
        if np.any((self.q_rate < 0) | (self.q_rate > 1)):
            raise ValueError(
                "Значения матрицы интенсивности поступления должны находиться в "
                "диапазоне [0, 1]."
            )
        # TODO: Спросить про корректность матриц интенсивности
