"""Модуль фабрики для создания вероятностных моделей СМО.

Функция `probability_system_factory` предоставляет интерфейс для инстанцирования
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
    MAPServerParams,
    MultiServerParams,
    SingleServerParams,
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
    params: Union[SingleServerParams, MultiServerParams, MAPServerParams],
    config: Union[ComputationConfig, MpmathComputationConfig, MergedComputationConfig],
) -> BaseProbabilitySystem:
    """Фабрика для создания системы массового обслуживания.

    Args:
        system_type (SystemType): Тип системы.
        calculation_method (CalculationMethod): Метод вычисления.
        params: Параметры системы соответствующего типа.
        config: Конфигурация вычислений.

    Returns:
        BaseProbabilitySystem: Инстанс соответствующей системы.
    """
    if calculation_method == CalculationMethod.ANALYTICAL:
        match system_type:
            case SystemType.SINGLE:
                return AnalyticalSingleServerSystem(params, config)
            case SystemType.MULTI:
                return AnalyticalMultiServerSystem(params, config)
            case SystemType.MAP:
                return AnalyticalMAPServerSystem(params, config)
            case _:
                raise ValueError(f"Неподдерживаемый тип системы: {system_type}")
    else:
        raise ValueError(f"Неподдерживаемый метод расчета: {calculation_method}")
