"""Модуль-фабрика для создания строителей матрицы моделей СМО.

Содержит функцию `matrix_builder_factory`, которая на основе переданных параметров
создаёт и возвращает соответствующий инстанс строителя матриц для различных типов СМО:
одноканальной, многоканальной или с MAP-потоками.
"""

from loguru import logger

from app.domain.models import MAPServerParams, MultiServerParams, SingleServerParams
from app.services.matrix_builders.base import BaseMatrixBuilder
from app.services.matrix_builders.map import MAPServerMatrixBuilder
from app.services.matrix_builders.multi import MultiServerMatrixBuilder
from app.services.matrix_builders.single import SingleServerMatrixBuilder


def matrix_builder_factory(
    params: SingleServerParams | MultiServerParams | MAPServerParams,
) -> BaseMatrixBuilder:
    """Фабрика для создания строителей матриц СМО по типу параметров.

    Args:
        params (SingleServerParams | MultiServerParams | MAPServerParams):
            Параметры конфигурации СМО.

    Returns:
        BaseMatrixBuilder: Соответствующий строитель матрицы.

    Raises:
        ValueError: При передаче неподдерживаемого типа параметров.
    """
    logger.debug(
        f"Вызван matrix_builder_factory для параметров: {type(params).__name__}"
    )

    if isinstance(params, SingleServerParams):
        logger.info("Создан SingleServerMatrixBuilder")
        return SingleServerMatrixBuilder(params)

    if isinstance(params, MultiServerParams):
        logger.info("Создан MultiServerMatrixBuilder")
        return MultiServerMatrixBuilder(params)

    if isinstance(params, MAPServerParams):
        logger.info("Создан MAPServerMatrixBuilder")
        return MAPServerMatrixBuilder(params)

    raise ValueError(f"Неподдерживаемый тип параметров: {type(params)}")
