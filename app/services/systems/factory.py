"""Модуль фабрики для создания СМО.

Функция `system_factory` предоставляет интерфейс для инстанцирования соответствующего
класса модели СМО в зависимости от типа системы и режима вычислений.
Поддерживается интеграция с конфигурациями вычислений различной точности.
"""

from typing import Any

from app.domain import CalculationMode, SystemType
from app.services.systems.base import BaseServerSystem
from app.services.systems.probability import ServerProbabilitySystem
from app.services.systems.throughput import (
    MAPServerThroughputSystem,
    ServerThroughputSystem,
)


def system_factory(
    system_type: SystemType, calculation_mode: CalculationMode, **kwargs: Any
) -> BaseServerSystem:
    """Фабрика для создания системы массового обслуживания.

    Args:
        system_type (SystemType): Тип системы.
        calculation_mode (CalculationMode): Режим вычисления.
        **kwargs (Any): Дополнительные параметры, передаваемые в конструктор системы
            (например: params, config).

    Returns:
        BaseServerSystem: Инстанс соответствующей системы.

    Raises:
        ValueError: Если передан неподдерживаемый режим расчёта или тип.
    """
    if calculation_mode == CalculationMode.PROBABILITY:
        return ServerProbabilitySystem(**kwargs)

    if calculation_mode == CalculationMode.THROUGHPUT:
        match system_type:
            case SystemType.MAP:
                return MAPServerThroughputSystem(**kwargs)
            case _:
                return ServerThroughputSystem(**kwargs)

    raise ValueError(f"Неподдерживаемый режим расчета: {calculation_mode}")
