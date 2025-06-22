"""Модуль фабрики для создания СМО.

Функция `system_factory` предоставляет интерфейс для инстанцирования соответствующего
класса модели СМО в зависимости от типа системы и режима вычислений.
Поддерживается интеграция с конфигурациями вычислений различной точности.
"""

from typing import Any

from app.domain import CalculationMode, SystemType
from app.services.systems.probability import (
    BaseProbabilitySystem,
    MAPServerSystem,
    MultiServerSystem,
    SingleServerSystem,
)
from app.services.systems.throughput import (
    BaseThroughputSystem,
    MAPServerThroughputSystem,
    MultiServerThroughputSystem,
    SingleServerThroughputSystem,
)


def system_factory(
    system_type: SystemType,
    calculation_mode: CalculationMode,
    **kwargs: Any,
) -> BaseProbabilitySystem | BaseThroughputSystem:
    """Фабрика для создания системы массового обслуживания.

    Args:
        system_type (SystemType): Тип системы.
        calculation_mode (CalculationMode): Режим вычисления.
        **kwargs (Any): Дополнительные параметры, передаваемые в конструктор системы
            (например: params, config).

    Returns:
        BaseProbabilitySystem | BaseThroughputSystem: Инстанс соответствующей системы.

    Raises:
        ValueError: Если передан неподдерживаемый режим расчёта или тип.
    """
    if calculation_mode == CalculationMode.PROBABILITY:
        match system_type:
            case SystemType.SINGLE:
                return SingleServerSystem(**kwargs)
            case SystemType.MULTI:
                return MultiServerSystem(**kwargs)
            case SystemType.MAP:
                return MAPServerSystem(**kwargs)
            case _:
                raise ValueError(f"Неподдерживаемый тип системы: {system_type}")
    if calculation_mode == CalculationMode.THROUGHPUT:
        match system_type:
            case SystemType.SINGLE:
                return SingleServerThroughputSystem(**kwargs)
            case SystemType.MULTI:
                return MultiServerThroughputSystem(**kwargs)
            case SystemType.MAP:
                return MAPServerThroughputSystem(**kwargs)
            case _:
                raise ValueError(f"Неподдерживаемый тип системы: {system_type}")
    else:
        raise ValueError(f"Неподдерживаемый режим расчета: {calculation_mode}")
