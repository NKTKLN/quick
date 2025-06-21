"""Модуль определения типов вычислений и конфигураций для численного моделирования СМО.

Содержит перечисления и dataclass для настройки параметров вычислений,
включая выбор метода, точность и параметры контроля погрешности.
"""

from dataclasses import dataclass
from enum import Flag, auto


class CalculationEngine(Flag):
    """Флаги движков вычислений, используемых в системе.

    Attributes:
        NUMPY: Стандартные вычисления с использованием библиотеки NumPy.
        MPMATH: Высокоточные вычисления с использованием библиотеки mpmath.
        MERGED: Комбинированный режим, объединяющий NumPy и mpmath.
    """

    NUMPY = auto()
    """Вычисления с использованием библиотеки NumPy (стандартная точность)."""

    MPMATH = auto()
    """Вычисления с использованием библиотеки mpmath (повышенная точность)."""

    MERGED = auto()
    """Комбинированный режим вычислений с применением NumPy и mpmath."""


class CalculationMode(Flag):
    """Флаги режимов вычислений, применяемых в системе.

    Позволяют указывать, какие метрики требуется рассчитывать.

    Attributes:
        PROBABILITY: Расчёт вероятностей состояний системы.
        THROUGHPUT: Расчёт пропускной способности системы.
    """

    PROBABILITY = auto()
    """Режим расчёта вероятностей состояний системы."""

    THROUGHPUT = auto()
    """Режим расчёта пропускной способности системы."""


class CalculationMethod(Flag):
    """Флаги методов расчёта математических моделей.

    Указывают, какой тип метода использовать для расчётов.

    Attributes:
        ANALYTICAL: Аналитический метод вычислений.
        NUMERICAL: Численный (приближённый) метод вычислений.
    """

    ANALYTICAL = auto()
    """Аналитический метод расчёта (точный, при наличии формул)."""

    NUMERICAL = auto()
    """Численный метод расчёта (приближённый)."""


class SystemType(Flag):
    """Флаги типов систем массового обслуживания (СМО), поддерживаемых системой.

    Используется для указания структуры обслуживающей системы.

    Attributes:
        SINGLE: Одноканальная СМО.
        MULTI: Многоканальная (многолинейная) СМО.
        MAP: СМО с MAP-потоками (марковский модулированный пуассоновский процесс).
    """

    SINGLE = auto()
    """Одноканальная система массового обслуживания."""

    MULTI = auto()
    """Многоканальная система массового обслуживания."""

    MAP = auto()
    """Система массового обслуживания с MAP-потоками."""


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
