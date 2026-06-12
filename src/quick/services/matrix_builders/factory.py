"""Модуль-фабрика для создания строителей матрицы моделей СМО.

Содержит функцию `matrix_builder_factory`, которая на основе переданных параметров
создаёт и возвращает соответствующий инстанс строителя матриц для различных типов СМО:
одноканальной, многоканальной или с Марковскими входными потоками.
"""

from loguru import logger

from quick.domain import SystemType
from quick.domain.params import SystemParams
from quick.services.matrix_builders.base import BaseMatrixBuilder
from quick.services.matrix_builders.map import MultiSensorMAPServerMatrixBuilder
from quick.services.matrix_builders.multi import MultiServerMatrixBuilder


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
        "Вызван matrix_builder_factory для типа системы: "
        f"{params.calculation_params.system_type}"
    )

    match params.calculation_params.system_type:
        case SystemType.MULTI:
            logger.info("Создан MultiServerMatrixBuilder")
            return MultiServerMatrixBuilder(params.base_params)
        case SystemType.MAP:
            logger.info("Создан MultiSensorMAPServerMatrixBuilder")
            return MultiSensorMAPServerMatrixBuilder(params.base_params)
        case _:
            logger.error(
                f"Неподдерживаемый тип системы: {params.calculation_params.system_type}"
            )
            raise ValueError(
                f"Неподдерживаемый тип системы: {params.calculation_params.system_type}"
            )
