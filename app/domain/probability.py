"""Модуль параметров систем массового обслуживания с нетерпеливыми заявками.

Содержит классы параметров для одно- и многолинейных систем массового обслуживания
с учетом интенсивности ухода нетерпеливых заявок (ν), включая модели с MAP-потоками.

Предоставляет структуры данных для моделирования переходных процессов и анализа
характеристик производительности системы при наличии оттока заявок.
"""

from dataclasses import dataclass

import numpy as np

from app.domain.base_params import (
    BasicMultiServerParams,
    BasicServerParams,
    BasicSingleServerParams,
)


@dataclass
class SingleServerParams(BasicSingleServerParams):
    """Параметры однолинейной СМО с учетом ухода нетерпеливых заявок.

    Используются для моделирования систем, в которых часть заявок может покинуть
    очередь до обслуживания с фиксированной интенсивностью ν.

    Attributes:
        nu_rate (float): Интенсивность ухода нетерпеливых заявок из очереди (ν ≥ 0).
        Остальные параметры наследуются из BasicSingleServerParams.
    """

    nu_rate: float

    def validate(self):
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
        super().validate()
        if self.nu_rate <= 0:
            raise ValueError("Интенсивность ν должна быть положительна.")
        if self.initial_probabilities.shape[0] != self.max_customers:
            raise ValueError(
                "Размер начальных вероятностей должен совпадать с максимальным числом "
                "заявок в системе."
            )


@dataclass
class MultiServerParams(BasicMultiServerParams):
    """Параметры многолинейной СМО с учетом ухода нетерпеливых заявок.

    Расширяет модель одноканальной системы с учетом множественных обслуживающих
    каналов (процессоров) и ухода заявок с фиксированной интенсивностью ν.

    Attributes:
        nu_rate (float): Интенсивность ухода нетерпеливых заявок из очереди (ν ≥ 0).
        Остальные параметры наследуются из BasicMultiServerParams.
    """

    nu_rate: float

    def validate(self):
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
        super().validate()
        if self.nu_rate <= 0:
            raise ValueError("Интенсивность ν должна быть положительна.")
        if (
            self.initial_probabilities.shape[0]
            != self.max_customers + self.processor_count + 1
        ):
            raise ValueError(
                "Размер начальных вероятностей должен совпадать с максимальным числом "
                "заявок в системе + колличество процессоров + 1."
            )


@dataclass
class MAPServerParams(BasicServerParams):
    """Параметры СМО с MAP-потоками и уходом нетерпеливых заявок.

    Описывает систему с марковским модулированным пуассоновским входным потоком,
    а также оттоком нетерпеливых заявок. Включает матрицы интенсивностей переходов.

    Attributes:
        nu_rate (float): Интенсивность ухода заявок (ν ≥ 0).
        p_rate (np.ndarray[np.float64]): Матрица интенсивностей обслуживания.
        q_rate (np.ndarray[np.float64]): Матрица интенсивностей поступления.
        Остальные параметры наследуются от BasicSingleServerParams.
    """  # TODO: Проверить корректность описания p_rate и q_rate

    nu_rate: float
    lambda_rate: np.ndarray[np.float64]
    p_rate: np.ndarray[np.float64]
    q_rate: np.ndarray[np.float64]

    def validate(self):
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
        super().validate()
        if self.nu_rate <= 0:
            raise ValueError("Интенсивность ν должна быть положительна.")
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
