"""Модуль определения типов вычислений и конфигураций для численного моделирования СМО.

Содержит перечисления и dataclass для настройки параметров вычислений, включая выбор
метода, точность и параметры контроля погрешности.
"""

from dataclasses import dataclass

from quick.domain.enums import CalculationEngine, CalculationMethod


@dataclass
class ComputationConfig:
    """Базовая конфигурация вычислений.

    Attributes:
        calculation_engine (CalculationEngine | None): Движок вычислений.
        calculation_method (CalculationMethod): Метод расчета.
        disable_cache (bool): Флаг отключения кэширования вычислений.
        compute_only_last_state (bool): Флаг вычисления только последнего состояния.
    """

    calculation_engine: CalculationEngine | None = None
    calculation_method: CalculationMethod = CalculationMethod.IMITATION
    disable_cache: bool = False
    compute_only_last_state: bool = False

    def validate(self) -> None:
        """Проверяет корректность параметров."""
        ...


@dataclass
class MpmathComputationConfig(ComputationConfig):
    """Конфигурация вычислений с повышенной точностью (mpmath).

    Attributes:
        precision (int): Точность вычислений - количество знаков после запятой
            (по умолчанию 50).
    """

    precision: int = 50

    def validate(self) -> None:
        """Проверяет корректность параметров.

        Raises:
            ValueError: При некорректных параметрах.
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

        Raises:
            ValueError: При некорректных параметрах.
        """
        super().validate()
        if self.tolerance <= 0:
            raise ValueError(
                "Допустимая погрешность должна находиться в диапазоне [0, 1]."
            )
