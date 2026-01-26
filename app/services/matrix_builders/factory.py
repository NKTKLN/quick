"""Модуль-фабрика для создания строителей матрицы моделей СМО.

Содержит функцию `matrix_builder_factory`, которая на основе переданных параметров
создаёт и возвращает соответствующий инстанс строителя матриц для различных типов СМО:
одноканальной, многоканальной или с MAP-потоками.
"""

from loguru import logger

from app.domain import SystemType
from app.domain.models import SystemParams
from app.services.matrix_builders.base import BaseMatrixBuilder
from app.services.matrix_builders.map import MAPServerMatrixBuilder
from app.services.matrix_builders.multi import MultiServerMatrixBuilder
from app.services.matrix_builders.multi_sensor_map import (
    MultiSensorMAPServerMatrixBuilder,
)


def matrix_builder_factory(params: SystemParams) -> BaseMatrixBuilder:
    """Фабрика для создания строителей матриц СМО по типу параметров.

    Args:
        params (ServerParams): Параметры конфигурации СМО.

    Returns:
        BaseMatrixBuilder: Соответствующий строитель матрицы.

    Raises:
        ValueError: При передаче неподдерживаемого типа параметров.
    """
    logger.debug(
        f"Вызван matrix_builder_factory для типа системы: {params.settings.system_type}"
    )

    match params.settings.system_type:
        case SystemType.MULTI:
            logger.info("Создан MultiServerMatrixBuilder")
            return MultiServerMatrixBuilder(params.base_params)
        case SystemType.MAP:
            logger.info("Создан MAPServerMatrixBuilder")
            return MAPServerMatrixBuilder(params.base_params)
        case SystemType.MULTI_SENSOR_MAP:
            logger.info("Создан MultiSensorMAPServerMatrixBuilder")
            return MultiSensorMAPServerMatrixBuilder(params.base_params)
        case _:
            raise ValueError(
                f"Неподдерживаемый тип системы: {params.settings.system_type}"
            )
