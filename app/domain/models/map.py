"""Модуль параметров систем массового обслуживания с MAP-потоками.

Содержит классы, описывающие параметры для СМО с марковским модулированным
пуассоновским входным потоком (MAP), возможностью ухода заявок и анализом
пропускной способности.
"""

from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray

from app.domain.iterators import ParamsNuIterator
from app.domain.models.base import BasicServerParams


@dataclass
class BasicMAPServerParams(BasicServerParams):
    """Параметры СМО с MAP-потоками и уходом нетерпеливых заявок.

    Описывает систему с марковским модулированным пуассоновским входным потоком,
    а также оттоком нетерпеливых заявок. Включает матрицы интенсивностей переходов.

    Attributes:
        lambda_rate (NDArray[np.float64]): Массив значений интенсивности
            поступления заявок (λ > 0).
        p_rate (NDArray[np.float64]): Матрица интенсивностей обслуживания.
        q_rate (NDArray[np.float64]): Матрица интенсивностей поступления.
        Остальные параметры наследуются от BasicSingleServerParams.
    """  # TODO: Проверить корректность описания p_rate и q_rate

    lambda_rate: NDArray[np.float64]
    p_rate: NDArray[np.float64]
    q_rate: NDArray[np.float64]

    def validate(self) -> None:
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


@dataclass
class MAPServerParams(BasicMAPServerParams):
    """Параметры СМО с MAP-потоками и уходом нетерпеливых заявок.

    Описывает систему с марковским модулированным пуассоновским входным потоком,
    а также оттоком нетерпеливых заявок. Включает матрицы интенсивностей переходов.

    Attributes:
        nu_rate (float): Интенсивность ухода заявок (ν ≥ 0).
        Остальные параметры наследуются от BasicMAPServerParams.
    """

    nu_rate: float

    def validate(self) -> None:
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
        super().validate()
        if self.nu_rate <= 0:
            raise ValueError("Интенсивность ν должна быть положительна.")


@dataclass
class MAPServerThroughputParams(BasicMAPServerParams):
    """Параметры СМО с MAP-потоками для анализа пропускной способности с перебором ν.

    Позволяет задавать массив значений интенсивности ухода заявок (ν) итерироваться
    по ним, создавая на каждой итерации объект BasicMAPServerParams.

    Attributes:
        nu_rate (NDArray[np.float64]): Массив значений интенсивности ухода заявок.
        Остальные параметры наследуются от BasicMAPServerParams.
    """

    nu_rate: NDArray[np.float64]

    def __iter__(self) -> ParamsNuIterator:
        """Создаёт итератор по значениям ν из массива nu_rate.

        Возвращает итератор, который на каждой итерации выдаёт
        MAPServerParams с текущим ν и базовыми параметрами.

        Returns:
            ParamsNuIterator: Итератор параметров СМО с разными значениями ν.
        """
        base_params = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "nu_rate"
        }
        return ParamsNuIterator(self.nu_rate, base_params, MAPServerParams)

    def validate(self) -> None:
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
        super().validate()
        if np.any(self.nu_rate <= 0) or self.nu_rate.shape[0] == 0:
            raise ValueError("Интенсивность ν должна быть положительна.")
