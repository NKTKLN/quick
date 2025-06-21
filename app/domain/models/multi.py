"""Модуль параметров многоканальных систем массового обслуживания (СМО).

Содержит классы параметров для многоканальных СМО без и с уходом заявок,
включая поддержку перебора интенсивности ухода заявок (ν).
"""

from dataclasses import dataclass, fields

import numpy as np

from app.domain.iterators import ParamsNuIterator
from app.domain.models.single import BasicSingleServerParams


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
class MultiServerThroughputParams(BasicMultiServerParams):
    """Параметры многолинейной СМО для анализа пропускной способности с перебором ν.

    Позволяет задавать массив значений интенсивности ухода заявок (ν) итерироваться
    по ним, создавая на каждой итерации объект MultiServerParams.

    Attributes:
        nu_rate (np.ndarray[np.float64]): Массив значений интенсивности ухода заявок.
        Остальные параметры наследуются от BasicMultiServerParams.
    """

    nu_rate: np.ndarray[np.float64]

    def __iter__(self) -> ParamsNuIterator:
        """Создаёт итератор по значениям ν из массива nu_rate.

        Возвращает итератор, который на каждой итерации выдаёт
        MultiServerParams с текущим ν и базовыми параметрами.

        Returns:
            ParamsNuIterator: Итератор параметров СМО с разными значениями ν.
        """
        base_params = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "nu_rate"
        }
        return ParamsNuIterator(self.nu_rate, base_params, MultiServerParams)

    def validate(self):
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
        super().validate()
        if np.any(self.nu_rate <= 0) or self.nu_rate.shape[0] == 0:
            raise ValueError("Интенсивность ν должна быть положительна.")
        if (
            self.initial_probabilities.shape[0]
            != self.max_customers + self.processor_count + 1
        ):
            raise ValueError(
                "Размер начальных вероятностей должен совпадать с максимальным числом "
                "заявок в системе + колличество процессоров + 1."
            )
