"""Модуль определения типов вычислений и конфигураций для численного моделирования СМО.

Содержит перечисления и dataclass для настройки параметров вычислений,
включая выбор метода, точность и параметры контроля погрешности.
"""

from dataclasses import dataclass
from enum import Flag, auto


class CalculationType(Flag):
    """Флаги типов вычислений, используемых в системе.

    Позволяют комбинировать методы вычислений через побитовые операции:
        - NUMPY: стандартные вычисления с помощью библиотеки NumPy
        - MPMATH: высокоточные вычисления с использованием библиотеки mpmath
        - MERGED: комбинированный режим, объединяющий NumPy и mpmath
    """

    NUMPY = auto()
    """Вычисления с использованием библиотеки NumPy (стандартная точность)."""

    MPMATH = auto()
    """Вычисления с использованием библиотеки mpmath (повышенная точность)."""

    MERGED = auto()
    """Комбинированный режим вычислений с применением NumPy и mpmath."""


@dataclass
class ComputationConfig:
    """Базовая конфигурация вычислений.

    Attributes:
        calculation_type (CalculationType): Тип вычислений для использования.
        disable_cache (bool): Флаг отключения кэширования вычислений.
    """

    calculation_type: CalculationType = CalculationType.MPMATH
    disable_cache: bool = False


@dataclass
class MpmathComputationConfig(ComputationConfig):
    """Конфигурация вычислений с повышенной точностью (mpmath).

    Attributes:
        precision (int): Точность вычислений — количество знаков после запятой
                         (по умолчанию 50).
    """

    precision: int = 50


@dataclass
class MergedComputationConfig(MpmathComputationConfig):
    """Конфигурация смешанного режима вычислений (NumPy + mpmath).

    Attributes:
        tolerance (float): Допустимая погрешность при сравнении результатов из
                           разных источников (по умолчанию 1e-12).
    """

    tolerance: float = 1e-12
