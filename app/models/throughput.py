"""Модуль параметров для анализа пропускной способности систем массового обслуживания.

Содержит классы расширенных параметров для одно- и многолинейных СМО с возможностью
итерации по массиву значений интенсивности ухода нетерпеливых заявок (ν),
что упрощает анализ чувствительности модели.
"""

from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray

from app.models.iterators import ParamsNuIterator
from app.models.probability import MultiServerParams, SingleServerParams


@dataclass
class SingleServerThroughputParams(SingleServerParams):
    """Параметры однолинейной СМО для анализа пропускной способности по массиву ν.

    Позволяет задать массив значений интенсивности ухода (ν) и организовать
    итерацию по ним с генерацией параметров для каждой интенсивности.

    Атрибуты:
        nu_rate (NDArray[np.float64]): Массив значений интенсивности ухода заявок (ν).
        Остальные параметры наследуются от BasicSingleServerParams.
    """

    nu_rate: NDArray[np.float64]  # Массив интенсивностей ухода заявок (ν)

    def __iter__(self) -> ParamsNuIterator:
        """Создаёт итератор по значениям ν из массива nu_rate.

        Итератор возвращает на каждой итерации экземпляр SingleServerParams
        с текущим значением ν и остальными параметрами.

        Возвращает:
            ParamsNuIterator: Итератор параметров СМО с разными значениями ν.
        """
        base_params = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "nu_rate"
        }
        return ParamsNuIterator(self.nu_rate, base_params, SingleServerParams)


@dataclass
class MultiServerThroughputParams(MultiServerParams):
    """Параметры многолинейной СМО для анализа пропускной способности по массиву ν.

    Позволяет задавать массив значений интенсивности ухода заявок (ν) итерироваться
    по ним, возвращая на каждой итерации экземпляр MultiServerParams.

    Атрибуты:
        nu_rate (NDArray[np.float64]): Массив значений интенсивности ухода заявок (ν).
        Остальные параметры наследуются от MultiServerParams.
    """

    nu_rate: NDArray[np.float64]  # Массив интенсивностей ухода заявок (ν)

    def __iter__(self) -> ParamsNuIterator:
        """Создаёт итератор по значениям ν из массива nu_rate.

        Итератор возвращает на каждой итерации экземпляр MultiServerParams
        с текущим значением ν и остальными параметрами.

        Возвращает:
            ParamsNuIterator: Итератор параметров СМО с разными значениями ν.
        """
        base_params = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "nu_rate"
        }
        return ParamsNuIterator(self.nu_rate, base_params, MultiServerParams)
