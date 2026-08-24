"""Модуль фабрики для создания экземпляров систем массового обслуживания (СМО).

Функция `system_factory` выбирает и создаёт подходящий класс модели СМО
в зависимости от типа системы и режима её работы.

Поддерживается интеграция с различными режимами расчётов (стационарный, переходный)
и типами систем (многоканальная, MAP и др.).
"""

from typing import Any

from loguru import logger

from quick.domain import SystemMode, SystemType
from quick.services.systems.base import BaseServerSystem
from quick.services.systems_behavior import (
    MultiSensorMAPSensorBehavior,
    MultiSensorMAPSystemBehavior,
    MultiSystemBehavior,
)
from quick.services.systems_behavior.base import BaseSystemBehavior

from .steady_state import MAPSteadyStateSystem, MultiServerSteadyStateSystem
from .transient_state import (
    TransientMAPServerStateSystem,
    TransientMultiServerStateSystem,
)


def system_factory(
    system_type: SystemType,
    system_mode: SystemMode,
    per_sensor: bool = False,
    **kwargs: Any,
) -> BaseServerSystem:
    """Фабрика для создания объекта системы массового обслуживания.

    Args:
        system_type (SystemType): Тип системы.
        system_mode (SystemMode): Режим работы системы.
        per_sensor (bool): Выполнять расчёт отдельно для каждого сенсора,
            возвращая массивы формы ``[время, датчик]``. Поддерживается
            только для систем типа MAP и игнорируется, если в
            ``params.calculation_params`` задан ``sensor_index``.
        **kwargs (Any): Дополнительные параметры, передаваемые в конструктор системы
            (например: params, config).

    Returns:
        BaseServerSystem: Экземпляр подходящего класса системы.

    Raises:
        ValueError: Если указан неподдерживаемый режим работы или тип системы.
        NotImplementedError: Если выбранный режим/тип ещё не реализован.
    """
    logger.debug(f"Вызван system_factory с параметрами: {system_type=}, {system_mode=}")

    system_behavior: type[BaseSystemBehavior]
    match system_type:
        case SystemType.MULTI:
            system_behavior = MultiSystemBehavior
        case SystemType.MAP:
            # Индивидуальный режим (расчёт по одному датчику) реализован
            # сужением оси MAP-фаз внутри агрегированного поведения, поэтому
            # он несовместим с покомпонентным разложением сразу по всем
            # датчикам и имеет над ним приоритет.
            individual_mode = (
                kwargs["params"].calculation_params.sensor_index is not None
            )
            system_behavior = (
                MultiSensorMAPSensorBehavior
                if per_sensor and not individual_mode
                else MultiSensorMAPSystemBehavior
            )

    if system_mode == SystemMode.TRANSIENT:
        match system_type:
            case SystemType.MULTI:
                logger.info("Создан TransientMultiServerStateSystem")
                return TransientMultiServerStateSystem(
                    **kwargs, system_behavior=system_behavior
                )
            case SystemType.MAP:
                logger.info("Создан TransientMAPServerStateSystem")
                return TransientMAPServerStateSystem(
                    **kwargs, system_behavior=system_behavior
                )

    if system_mode == SystemMode.STEADY:
        match system_type:
            case SystemType.MULTI:
                logger.info("Создан MultiServerSteadyStateSystem")
                return MultiServerSteadyStateSystem(
                    **kwargs, system_behavior=system_behavior
                )
            case SystemType.MAP:
                logger.info("Создан MAPSteadyStateSystem")
                return MAPSteadyStateSystem(**kwargs, system_behavior=system_behavior)
