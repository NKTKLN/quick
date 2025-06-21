"""Модуль параметров одноканальных систем массового обслуживания (СМО).

Содержит классы параметров для одноканальных СМО без учета ухода заявок,
с фиксированной интенсивностью ухода, а также для перебора интенсивности ухода (ν).
"""

from dataclasses import dataclass, fields

import numpy as np

from app.domain.iterators import ParamsNuIterator
from app.domain.models.base import BasicServerParams


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
class SingleServerThroughputParams(BasicSingleServerParams):
    """Параметры однолинейной СМО для анализа пропускной способности с перебором ν.

    Позволяет задавать массив значений интенсивности ухода заявок (ν) итерироваться
    по ним, создавая на каждой итерации объект SingleServerParams.

    Attributes:
        nu_rate (np.ndarray[np.float64]): Массив значений интенсивности ухода заявок.
        Остальные параметры наследуются от BasicSingleServerParams.
    """

    nu_rate: np.ndarray[np.float64]

    def __iter__(self) -> ParamsNuIterator:
        """Создаёт итератор по значениям ν из массива nu_rate.

        Возвращает итератор, который на каждой итерации выдаёт
        SingleServerParams с текущим ν и базовыми параметрами.

        Returns:
            ParamsNuIterator: Итератор параметров СМО с разными значениями ν.
        """
        base_params = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "nu_rate"
        }
        return ParamsNuIterator(self.nu_rate, base_params, SingleServerParams)

    def validate(self):
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
        super().validate()
        if np.any(self.nu_rate <= 0) or self.nu_rate.shape[0] == 0:
            raise ValueError("Интенсивность ν должна быть положительна.")
        if self.initial_probabilities.shape[0] != self.max_customers:
            raise ValueError(
                "Размер начальных вероятностей должен совпадать с максимальным числом "
                "заявок в системе."
            )
