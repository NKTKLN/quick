"""Модуль фабрики для пропускной способности СМО.

Функция `throughput_system_factory` предоставляет интерфейс для инстанцирования
соответствующего класса модели СМО в зависимости от типа системы и метода вычислений.
Поддерживается интеграция с конфигурациями вычислений различной точности.
"""

from typing import Union

from app.domain import (
    CalculationMethod,
    ComputationConfig,
    MergedComputationConfig,
    MpmathComputationConfig,
    SystemType,
)
from app.domain.models import (
    MAPServerThroughputParams,
    MultiServerThroughputParams,
    SingleServerThroughputParams,
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
    params: Union[
        SingleServerThroughputParams,
        MultiServerThroughputParams,
        MAPServerThroughputParams,
    ],
    config: Union[ComputationConfig, MpmathComputationConfig, MergedComputationConfig],
) -> BaseThroughputSystem:
    """Фабрика для создания системы массового обслуживания.

    Args:
        system_type (SystemType): Тип системы.
        calculation_method (CalculationMethod): Метод вычисления.
        params: Параметры системы соответствующего типа.
        config: Конфигурация вычислений.

    Returns:
        BaseThroughputSystem: Инстанс соответствующей системы.
    """
    if calculation_method == CalculationMethod.ANALYTICAL:
        match system_type:
            case SystemType.SINGLE:
                return AnalyticalSingleServerThroughputSystem(params, config)
            case SystemType.MULTI:
                return AnalyticalMultiServerThroughputSystem(params, config)
            case SystemType.MAP:
                return AnalyticalMAPServerThroughputSystem(params, config)
            case _:
                raise ValueError(f"Неподдерживаемый тип системы: {system_type}")
    else:
        raise ValueError(f"Неподдерживаемый метод расчета: {calculation_method}")
