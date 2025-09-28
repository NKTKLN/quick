"""Модуль фабрики для создания экземпляров систем массового обслуживания (СМО).

Функция `system_factory` выбирает и создаёт подходящий класс модели СМО
в зависимости от типа системы и режима её работы.

Поддерживается интеграция с различными режимами расчётов (стационарный, переходный)
и типами систем (многоканальная, MAP и др.).
"""

from typing import Any

from loguru import logger

from app.domain import SystemMode, SystemType
from app.services.systems.base import BaseServerSystem
from app.services.systems.steady_state import MultiServerSteadyStateSystem
from app.services.systems.transient_state import TransientServerStateSystem
from app.services.systems_behavior import (
    MAPSystemBehavior,
    MultiSystemBehavior,
)


def system_factory(
    system_type: SystemType, system_mode: SystemMode, **kwargs: Any
) -> BaseServerSystem:
    """Фабрика для создания объекта системы массового обслуживания.

    Args:
        system_type (SystemType): Тип системы.
        system_mode (SystemMode): Режим работы системы.
        **kwargs (Any): Дополнительные параметры, передаваемые в конструктор системы
            (например: params, config).

    Returns:
        BaseServerSystem: Экземпляр подходящего класса системы.

    Raises:
        ValueError: Если указан неподдерживаемый режим работы или тип системы.
        NotImplementedError: Если выбранный режим/тип ещё не реализован.
    """
    logger.debug(f"Вызван system_factory с параметрами: {system_type=}, {system_mode=}")

    system_behavior = None
    match system_type:
        case SystemType.MULTI:
            system_behavior = MultiSystemBehavior
        case SystemType.MAP:
            system_behavior = MAPSystemBehavior
        case _:
            raise NotImplementedError(f"Система типа {system_type} не реализована")

    if system_mode == SystemMode.TRANSIENT:
        logger.info("Создан TransientServerStateSystem")
        return TransientServerStateSystem(**kwargs, system_behavior=system_behavior)

    if system_mode == SystemMode.STEADY:
        match system_type:
            case SystemType.MULTI:
                logger.info("Создан MultiServerSteadyStateSystem")
                return MultiServerSteadyStateSystem(
                    **kwargs, system_behavior=system_behavior
                )
            case _:
                raise NotImplementedError(
                    f"Система типа {system_type} в режиме STEADY не реализована"
                )

    raise ValueError(f"Неподдерживаемый режим функционирования: {system_mode}")
