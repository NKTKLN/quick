"""Модуль-фабрика для создания строителей матрицы моделей СМО.

Содержит функцию `matrix_builder_factory`, которая на основе переданных параметров
создаёт и возвращает соответствующий инстанс строителя матриц для различных типов СМО:
одноканальной, многоканальной или с MAP-потоками.
"""

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
    if isinstance(params, SingleServerParams):
        return SingleServerMatrixBuilder(params)
    elif isinstance(params, MultiServerParams):
        return MultiServerMatrixBuilder(params)
    elif isinstance(params, MAPServerParams):
        return MAPServerMatrixBuilder(params)
    else:
        raise ValueError(f"Unsupported parameter type: {type(params)}")
