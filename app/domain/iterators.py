"""Итератор параметров СМО с различными значениями интенсивности ухода.

Содержит класс ParamsNuIterator, который позволяет поочередно создавать
экземпляры параметров системы массового обслуживания (СМО) с разными
значениями интенсивности ухода нетерпеливых заявок (ν), перебирая значения
из заданного массива.

Класс:
    ParamsNuIterator: Итератор для генерации параметров СМО с разными ν.
"""

from typing import Any, Iterator, Type

import numpy as np
from numpy.typing import NDArray


class ParamsNuIterator(Iterator):
    """Итератор для последовательного получения параметров СМО с разными значениями ν.

    Позволяет проходить по массиву значений интенсивности ухода заявок (ν)
    и создавать на каждой итерации экземпляр класса параметров СМО с текущим ν.

    Атрибуты:
        nu_rate (NDArray[np.float64]): Массив значений интенсивности ухода заявок.
        base_params (dict[str, Any]): Базовые параметры СМО без ν.
        class_type (Type): Тип класса параметров для создания экземпляров.
        __position (int): Текущая позиция в массиве nu_rate.
    """

    def __init__(
        self,
        nu_rate: NDArray[np.float64],
        base_params: dict[str, Any],
        class_type: Type,
    ) -> None:
        """Инициализация итератора.

        Атрибуты:
            nu_rate (NDArray[np.float64]): Массив значений интенсивности ухода заявок.
            base_params (dict[str, Any]): Базовые параметры СМО (без ν).
            class_type (Type): Класс параметров СМО для создания экземпляров.
        """
        self.nu_rate = nu_rate
        self.base_params = base_params
        self.class_type = class_type
        self.__position = 0

    def __iter__(self) -> Iterator:
        """Возвращает сам итератор.

        Возвращает:
            Iterator: Сам объект итератора.
        """
        return self

    def __next__(self) -> Any:
        """Возвращает следующий набор параметров СМО с очередным значением ν.

        Возвращает:
            Экземпляр класса параметров СМО с текущим значением ν.

        Исключения:
            StopIteration: Когда достигнут конец массива nu_rate.
        """
        if self.__position < len(self.nu_rate):
            nu = self.nu_rate[self.__position]
            self.__position += 1
            return self.class_type(**self.base_params, nu_rate=nu)
        raise StopIteration
