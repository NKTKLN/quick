"""Модуль определения типов вычислений и конфигураций для численного моделирования СМО.

Содержит перечисления и dataclass для настройки параметров вычислений,
включая выбор метода, точность и параметры контроля погрешности.
"""

from dataclasses import dataclass
from enum import Flag, auto


class CalculationEngine(Flag):
    """Флаги движков вычислений, используемых в системе.

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


class CalculationMode(Flag):
    """Флаги режимов вычисления."""

    PROBABILITY = auto()
    THROUGHPUT = auto()


@dataclass
class ComputationConfig:
    """Базовая конфигурация вычислений.

    Attributes:
        calculation_engine (CalculationEngine): Движок вычислений.
        calculation_mode (CalculationMode): Режим вычисления.
        disable_cache (bool): Флаг отключения кэширования вычислений.
    """

    calculation_engine: CalculationEngine = CalculationEngine.MPMATH
    calculation_mode: CalculationMode = CalculationMode.PROBABILITY
    disable_cache: bool = False


@dataclass
class MpmathComputationConfig(ComputationConfig):
    """Конфигурация вычислений с повышенной точностью (mpmath).

    Attributes:
        precision (int): Точность вычислений — количество знаков после запятой
            (по умолчанию 50).
    """

    precision: int = 50

    def validate(self):
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
        if self.precision <= 0:
            raise ValueError("Точность вычислений должна быть положительна.")


@dataclass
class MergedComputationConfig(MpmathComputationConfig):
    """Конфигурация смешанного режима вычислений (NumPy + mpmath).

    Attributes:
        tolerance (float): Допустимая погрешность при сравнении результатов из
                           разных источников (по умолчанию 1e-12).
    """

    tolerance: float = 1e-12

    def validate(self):
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
        if self.tolerance <= 0:
            raise ValueError(
                "Допустимая погрешность должна находиться в диапазоне [0, 1]."
            )
