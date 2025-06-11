"""Модуль, определяющий типы вычислений и конфигурации для численного моделирования СМО.

Содержит классы перечисления и dataclass-ов для настройки параметров вычислений,
включая точность, используемый метод и допустимую погрешность.
"""

from dataclasses import dataclass
from enum import Flag, auto


class CalculationType(Flag):
    """Перечисление, определяющее типы вычислений, которые используются в системе.

    Поддерживает комбинации через битовые флаги:
        - NUMPY: использование библиотеки NumPy для стандартных вычислений
        - MPMATH: использование библиотеки mpmath для высокоточных вычислений
        - MERGED: смешанный режим вычислений (NumPy + mpmath)
    """

    NUMPY = auto()
    """Использование NumPy для стандартной точности."""

    MPMATH = auto()
    """Использование mpmath для повышенной точности."""

    MERGED = auto()
    """Смешанный режим: сочетание NumPy и mpmath."""


@dataclass
class ComputationConfig:
    """Базовая конфигурация вычислений.

    Attributes:
        calculation_type (CalculationType): Тип вычислений, который будет использоваться
    """

    calculation_type: CalculationType = CalculationType.MPMATH


@dataclass
class MpmathComputationConfig(ComputationConfig):
    """Конфигурация вычислений с повышенной точностью (mpmath).

    Attributes:
        precision (int): Количество знаков после запятой, используемых в вычислениях.
                         По умолчанию — 50.
    """

    precision: int = 50  # Количество знаков после запятой


@dataclass
class MergedComputationConfig(MpmathComputationConfig):
    """Конфигурация для смешанного режима вычислений (NumPy + mpmath).

    Attributes:
        tolerance (float): Допустимая погрешность при сравнении результатов из разных
                           источников. По умолчанию — 1e-16.
    """

    tolerance: float = 1e-16
