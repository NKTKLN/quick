"""Модуль фабрики для создания моделей СМО.

Функция `model_factory` предоставляет интерфейс для инстанцирования соответствующей
модели СМО в зависимости от типа системы и режима вычислений.
"""

from typing import Any

from loguru import logger

from app.domain import SystemType
from app.domain.models.base import BasicServerParams
from app.domain.models.map import MAPServerParams
from app.domain.models.multi import MultiServerParams
from app.domain.models.single import SingleServerParams


def model_factory(
    system_type: SystemType,
    **kwargs: Any,
) -> BasicServerParams:
    """Фабрика для создания моделей СМО.

    Args:
        system_type (SystemType): Тип системы.
        **kwargs (Any): Дополнительные параметры, передаваемые в конструктор системы
            (например: params, config).

    Returns:
        BasicServerParams: Инстанс соответствующей модели.

    Raises:
        ValueError: Если передан неподдерживаемый тип.
    """
    logger.debug(f"Вызван model_factory с параметром system_type={system_type}")

    match system_type:
        case SystemType.SINGLE:
            logger.info("Создан SingleServerParams")
            return SingleServerParams(**kwargs)
        case SystemType.MULTI:
            logger.info("Создан MultiServerParams")
            return MultiServerParams(**kwargs)
        case SystemType.MAP:
            logger.info("Создан MAPServerParams")
            return MAPServerParams(**kwargs)

    raise ValueError(f"Неподдерживаемый тип системы: {system_type}")
