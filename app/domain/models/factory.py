"""Модуль фабрики для создания моделей СМО.

Функция `model_factory` предоставляет интерфейс для инстанцирования соответствующей
модели СМО в зависимости от типа системы и режима вычислений.
"""

from typing import Any

from app.domain import CalculationMode, SystemType
from app.domain.models.base import BasicServerParams
from app.domain.models.map import (
    MAPServerParams,
    MAPServerThroughputParams,
)
from app.domain.models.multi import (
    MultiServerParams,
    MultiServerThroughputParams,
)
from app.domain.models.single import (
    SingleServerParams,
    SingleServerThroughputParams,
)


def model_factory(
    system_type: SystemType,
    calculation_mode: CalculationMode,
    **kwargs: Any,
) -> BasicServerParams:
    """Фабрика для создания моделей СМО.

    Args:
        system_type (SystemType): Тип системы.
        calculation_mode (CalculationMode): Режим вычисления.
        **kwargs (Any): Дополнительные параметры, передаваемые в конструктор системы
            (например: params, config).

    Returns:
        BasicServerParams: Инстанс соответствующей модели.

    Raises:
        ValueError: Если передан неподдерживаемый режим расчёта или тип.
    """
    if calculation_mode == CalculationMode.PROBABILITY:
        match system_type:
            case SystemType.SINGLE:
                return SingleServerParams(**kwargs)
            case SystemType.MULTI:
                return MultiServerParams(**kwargs)
            case SystemType.MAP:
                return MAPServerParams(**kwargs)
            case _:
                raise ValueError(f"Неподдерживаемый тип системы: {system_type}")
    if calculation_mode == CalculationMode.THROUGHPUT:
        match system_type:
            case SystemType.SINGLE:
                return SingleServerThroughputParams(**kwargs)
            case SystemType.MULTI:
                return MultiServerThroughputParams(**kwargs)
            case SystemType.MAP:
                return MAPServerThroughputParams(**kwargs)
            case _:
                raise ValueError(f"Неподдерживаемый тип системы: {system_type}")
    else:
        raise ValueError(f"Неподдерживаемый режим расчета: {calculation_mode}")
