"""Модуль параметров систем массового обслуживания с Марковскими входными потоками.

Реализует класс MAPSystemParams, который описывает параметры системы массового
обслуживания с марковским модулированным пуассоновским входным потоком (MAP) и
возможностью ухода заявок.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .base import BaseSystemParams


@dataclass
class MAPSystemParams(BaseSystemParams):
    """Базовые параметры СМО с Марковскими входными потоками и уходом нетерпеливых заявок.

    Описывает систему с марковским модулированным пуассоновским входным потоком,
    а также оттоком нетерпеливых заявок. Включает матрицы интенсивностей переходов.

    Attributes:
        p_rate (NDArray[np.float64]): Матрица интенсивностей обслуживания.
        q_rate (NDArray[np.float64]): Матрица интенсивностей поступления.
        sensor_count (int): Количество сенсоров в системе (m).
        Остальные параметры наследуются от BaseSystemParams.
    """

    p_rate: NDArray[np.float64]
    q_rate: NDArray[np.float64]
    sensor_count: int

    def validate(self) -> None:
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах. # TODO
        """
        super().validate()
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
        if np.sum(self.q_rate) == 0 or np.sum(self.q_rate) == 0:  # TODO
            raise ValueError(
                "Все значения матрицы интенсивности поступления или "
                "обслуживания не должны равняться нулю."
            )
        # TODO: Спросить про корректность матриц интенсивности (Пока не надо)
