"""Модуль параметров для анализа пропускной способности систем массового обслуживания.

Содержит классы расширенных параметров для одно- и многолинейных СМО с возможностью
итерации по массиву значений интенсивности ухода нетерпеливых заявок (ν),
что упрощает анализ чувствительности модели.
"""

from dataclasses import dataclass, fields

import numpy as np

from app.domain.iterators import ParamsNuIterator
from app.domain.probability import (
    BasicMultiServerParams,
    BasicSingleServerParams,
    MultiServerParams,
    SingleServerParams,
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
        if np.any(self.nu_rate <= 0):
            raise ValueError("Интенсивность ν должна быть положительна.")
        if self.initial_probabilities.shape[0] != self.max_customers:
            raise ValueError(
                "Размер начальных вероятностей должен совпадать с максимальным числом "
                "заявок в системе."
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
        if np.any(self.nu_rate <= 0):
            raise ValueError("Интенсивность ν должна быть положительна.")
        if (
            self.initial_probabilities.shape[0]
            != self.max_customers + self.processor_count
        ):
            raise ValueError(
                "Размер начальных вероятностей должен совпадать с максимальным числом "
                "заявок в системе + колличество процессоров."
            )
