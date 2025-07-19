"""Модуль фабрики для создания экземпляров систем массового обслуживания (СМО).

Функция `system_factory` выбирает и создаёт подходящий класс модели СМО
в зависимости от типа системы и режима её работы.

Поддерживается интеграция с различными режимами расчётов (стационарный, переходный)
и типами систем (многоканальная, MAP и др.).
"""

from typing import Any

from loguru import logger

from app.domain import SystemMode, SystemType
from app.services.systems.base import BaseSystem
from app.services.systems.steady_state import MultiServerSteadyStateSystem
from app.services.systems.transient_state import TransientStateSystem


def system_factory(
    system_type: SystemType, system_mode: SystemMode, **kwargs: Any
) -> BaseSystem:
    """Фабрика для создания объекта системы массового обслуживания.

    Args:
        system_type (SystemType): Тип системы.
        system_mode (SystemMode): Режим работы системы.
        **kwargs (Any): Дополнительные параметры, передаваемые в конструктор системы
            (например: params, config).

    Returns:
        BaseSystem: Экземпляр подходящего класса системы.

    Raises:
        ValueError: Если указан неподдерживаемый режим работы или тип системы.
        NotImplementedError: Если выбранный режим/тип ещё не реализован.
    """
    logger.debug(
        f"Вызван system_factory с параметрами: system_type={system_type}, "
        f"system_mode={system_mode}"
    )

    if system_mode == SystemMode.TRANSIENT:
        match system_type:
            case SystemType.MAP:
                # logger.info("Создан MAPServerThroughputSystem")
                raise NotImplementedError(
                    "MAP-системы в переходном режиме не реализованы"
                )
            case _:
                logger.info("Создан TransientStateSystem")
                return TransientStateSystem(**kwargs)

    if system_mode == SystemMode.STEADY:
        match system_type:
            case SystemType.MULTI:
                logger.info("Создан MultiServerSteadyStateSystem")
                return MultiServerSteadyStateSystem()
            case _:
                raise NotImplementedError(
                    f"Система типа {system_type} в режиме STEADY не реализована"
                )

    raise ValueError(f"Неподдерживаемый режим функционирования: {system_mode}")
