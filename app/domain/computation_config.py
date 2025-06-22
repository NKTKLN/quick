"""Модуль определения типов вычислений и конфигураций для численного моделирования СМО.

Содержит перечисления и dataclass для настройки параметров вычислений,
включая выбор метода, точность и параметры контроля погрешности.
"""

from dataclasses import dataclass

from app.domain.enums import CalculationEngine, CalculationMethod, CalculationMode


@dataclass
class ComputationConfig:
    """Базовая конфигурация вычислений.

    Attributes:
        calculation_engine (CalculationEngine): Движок вычислений.
        calculation_mode (CalculationMode): Режим вычисления.
        calculation_method (CalculationMethod): Метод расчета.
        disable_cache (bool): Флаг отключения кэширования вычислений.
    """

    calculation_engine: CalculationEngine = CalculationEngine.MPMATH
    calculation_mode: CalculationMode = CalculationMode.PROBABILITY
    calculation_method: CalculationMethod = CalculationMethod.ANALYTICAL
    disable_cache: bool = False

    def validate(self) -> None:
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
        pass


@dataclass
class MpmathComputationConfig(ComputationConfig):
    """Конфигурация вычислений с повышенной точностью (mpmath).

    Attributes:
        precision (int): Точность вычислений — количество знаков после запятой
            (по умолчанию 50).
    """

    precision: int = 50

    def validate(self) -> None:
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
        super().validate()
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

    def validate(self) -> None:
        """Проверяет корректность параметров.

        Выбрасывает исключение ValueError при некорректных параметрах.
        """
        super().validate()
        if self.tolerance <= 0:
            raise ValueError(
                "Допустимая погрешность должна находиться в диапазоне [0, 1]."
            )
