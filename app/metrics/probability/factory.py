"""Модуль фабрики для создания вероятностных моделей СМО.

Функция `probability_system_factory` предоставляет интерфейс для инстанцирования
соответствующего класса модели СМО в зависимости от типа системы и метода вычислений.
Поддерживается интеграция с конфигурациями вычислений различной точности.
"""

from typing import Any

from app.domain import (
    CalculationMethod,
    SystemType,
)
from app.metrics.probability.analytical import (
    AnalyticalMAPServerSystem,
    AnalyticalMultiServerSystem,
    AnalyticalSingleServerSystem,
)
from app.metrics.probability.base import BaseProbabilitySystem


def probability_system_factory(
    system_type: SystemType,
    calculation_method: CalculationMethod,
    **kwargs: Any,
) -> BaseProbabilitySystem:
    """Фабрика для создания системы массового обслуживания.

    Args:
        system_type (SystemType): Тип системы.
        calculation_method (CalculationMethod): Метод вычисления.
        **kwargs (Any): Дополнительные параметры, передаваемые в конструктор системы
            (например: params, config).

    Returns:
        BaseProbabilitySystem: Инстанс соответствующей системы.

    Raises:
        ValueError: Если передан неподдерживаемый метод расчёта или тип.
    """
    if calculation_method == CalculationMethod.ANALYTICAL:
        match system_type:
            case SystemType.SINGLE:
                return AnalyticalSingleServerSystem(**kwargs)
            case SystemType.MULTI:
                return AnalyticalMultiServerSystem(**kwargs)
            case SystemType.MAP:
                return AnalyticalMAPServerSystem(**kwargs)
            case _:
                raise ValueError(f"Неподдерживаемый тип системы: {system_type}")
    else:
        raise ValueError(f"Неподдерживаемый метод расчета: {calculation_method}")
