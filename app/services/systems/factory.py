"""Модуль фабрики для создания СМО.

Функция `system_factory` предоставляет интерфейс для инстанцирования соответствующего
класса модели СМО в зависимости от типа системы и режима вычислений.
Поддерживается интеграция с конфигурациями вычислений различной точности.
"""

from typing import Any

from loguru import logger

from app.domain import CalculationMode, SystemType
from app.services.systems.absolute_throughput import ServerAbsoluteThroughputSystem
from app.services.systems.base import BaseServerSystem
from app.services.systems.probability import ServerProbabilitySystem
from app.services.systems.relative_throughput import ServerRelativeThroughputSystem
from app.services.systems.throughput import (
    MAPServerThroughputSystem,
    ServerThroughputSystem,
)


# pylint: disable=too-many-return-statements
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
    logger.debug(
        f"Вызван system_factory с параметрами: system_type={system_type}, "
        f"calculation_mode={calculation_mode}"
    )

    if calculation_mode == CalculationMode.PROBABILITY:
        logger.info("Создан ServerProbabilitySystem")
        return ServerProbabilitySystem(**kwargs)

    if calculation_mode == CalculationMode.THROUGHPUT:
        match system_type:
            case SystemType.MAP:
                logger.info("Создан MAPServerThroughputSystem")
                return MAPServerThroughputSystem(**kwargs)
            case _:
                logger.info("Создан ServerThroughputSystem")
                return ServerThroughputSystem(**kwargs)

    if calculation_mode == CalculationMode.ABSOLUTE_THROUGHPUT:
        logger.info("Создан ServerAbsoluteThroughputSystem")
        return ServerAbsoluteThroughputSystem(**kwargs)

    if calculation_mode == CalculationMode.RELATIVE_THROUGHPUT:
        logger.info("Создан ServerRelativeThroughputSystem")
        return ServerRelativeThroughputSystem(**kwargs)

    raise ValueError(f"Неподдерживаемый режим расчета: {calculation_mode}")
