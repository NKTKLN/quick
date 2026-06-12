"""Фабрика имитационных решателей СМО."""

from loguru import logger

from quick.domain import SystemType
from quick.domain.params import ImitationSystemParams
from quick.services.solvers.base import BasicProbabilitySolver

from .map import MAPImitationProbabilitySolver
from .multi import MultiServerImitationProbabilitySolver


def imitation_solvers_factory(
    params: ImitationSystemParams,
) -> BasicProbabilitySolver:
    """Создаёт имитационный решатель СМО.

    Args:
        params: Параметры имитационной модели.

    Returns:
        Экземпляр имитационного решателя.

    Raises:
        ValueError: Если указан неподдерживаемый тип системы.
    """
    match params.calculation_params.system_type:
        case SystemType.MULTI:
            logger.info("Создан MultiServerImitationProbabilitySolver")
            return MultiServerImitationProbabilitySolver(params)

        case SystemType.MAP:
            logger.info("Создан MAPImitationProbabilitySolver")
            return MAPImitationProbabilitySolver(params)
