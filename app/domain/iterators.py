"""Итератор параметров СМО с различными значениями интенсивности ухода.

Позволяет генерировать последовательность экземпляров конфигураций СМО,
различающихся по параметру ν (интенсивность ухода заявок).
"""

from typing import Any, Callable, Iterator, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")  # Обобщённый тип для возвращаемых экземпляров


class ParamsNuIterator(Iterator[T]):
    """Итератор параметров СМО с различными значениями интенсивности ухода (ν)."""

    def __init__(
        self,
        nu_rate: NDArray[np.float64],
        base_params: dict[str, Any],
        class_type: Callable[..., T],
    ) -> None:
        """Инициализация итератора параметров.

        Args:
            nu_rate (NDArray[np.float64]): Массив значений ν.
            base_params (dict[str, Any]): Общие параметры СМО без ν.
            class_type (Callable[..., T]): Класс параметров СМО.
        """
        self.nu_rate = nu_rate
        self.base_params = base_params
        self.class_type = class_type
        self.__position = 0

    def __iter__(self) -> Iterator[T]:
        """Возвращает сам итератор."""
        return self

    def __next__(self) -> T:
        """Создаёт и возвращает следующий объект параметров с текущим ν.

        Returns:
            T: Экземпляр класса с установленным значением ν.

        Raises:
            StopIteration: Если достигнут конец массива ν.
        """
        if self.__position < len(self.nu_rate):
            nu = self.nu_rate[self.__position]
            self.__position += 1
            return self.class_type(**self.base_params, nu_rate=nu)
        raise StopIteration
