"""Модуль фабрики для пропускной способности СМО.

Функция `throughput_system_factory` предоставляет интерфейс для инстанцирования
соответствующего класса модели СМО в зависимости от типа системы и метода вычислений.
Поддерживается интеграция с конфигурациями вычислений различной точности.
"""

from typing import Any

from app.domain import (
    CalculationMethod,
    SystemType,
)
from app.metrics.throughput.analytical import (
    AnalyticalMAPServerThroughputSystem,
    AnalyticalMultiServerThroughputSystem,
    AnalyticalSingleServerThroughputSystem,
)
from app.metrics.throughput.base import BaseThroughputSystem


def throughput_system_factory(
    system_type: SystemType,
    calculation_method: CalculationMethod,
    **kwargs: Any,
) -> BaseThroughputSystem:
    """Фабрика для создания системы массового обслуживания.

    Args:
        system_type (SystemType): Тип системы.
        calculation_method (CalculationMethod): Метод вычисления.
        **kwargs (Any): Дополнительные параметры, передаваемые в конструктор системы
            (например: params, config).

    Returns:
        BaseThroughputSystem: Инстанс соответствующей системы.

    Raises:
        ValueError: Если передан неподдерживаемый метод расчёта или тип.
    """
    if calculation_method == CalculationMethod.ANALYTICAL:
        match system_type:
            case SystemType.SINGLE:
                return AnalyticalSingleServerThroughputSystem(**kwargs)
            case SystemType.MULTI:
                return AnalyticalMultiServerThroughputSystem(**kwargs)
            case SystemType.MAP:
                return AnalyticalMAPServerThroughputSystem(**kwargs)
            case _:
                raise ValueError(f"Неподдерживаемый тип системы: {system_type}")
    else:
        raise ValueError(f"Неподдерживаемый метод расчета: {calculation_method}")
