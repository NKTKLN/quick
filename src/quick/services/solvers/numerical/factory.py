"""Фабрика численных решателей СМО."""

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.domain import ComputationConfig
from quick.domain.params import TransientSystemParams
from quick.services.solvers.base import BasicProbabilitySolver

from .solver import NumericalProbabilitySolver


def numerical_solvers_factory(
    transient_params: TransientSystemParams,
    coefficients_matrix: NDArray[np.float64],
    config: ComputationConfig,
) -> BasicProbabilitySolver:
    """Создаёт численный решатель СМО.

    Args:
        transient_params: Переходные параметры системы.
        coefficients_matrix: Квадратная матрица коэффициентов (n x n).
        config: Конфигурация вычислений.

    Returns:
        Экземпляр численного решателя.
    """
    logger.info("Создан NumericalProbabilitySolver")
    return NumericalProbabilitySolver(
        transient_params,
        coefficients_matrix,
        config,
    )
