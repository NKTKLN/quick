"""Фабрика для создания аналитических решателей вероятностной модели СМО."""

from typing import cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.domain import CalculationEngine, ComputationConfig
from quick.domain.computation_config import (
    MergedComputationConfig,
    MpmathComputationConfig,
)
from quick.domain.models import TransientSystemParams
from quick.services.solvers.base import BasicProbabilitySolver

from .merged_solver import AnalyticalMergedProbabilitySolver
from .mpmath_solver import AnalyticalMpmathProbabilitySolver
from .numpy_solver import AnalyticalNumpyProbabilitySolver


def analytical_solvers_factory(
    transient_params: TransientSystemParams,
    coefficients_matrix: NDArray[np.float64],
    config: ComputationConfig,
) -> BasicProbabilitySolver:
    """Создаёт аналитический решатель СМО.

    Args:
        transient_params: Переходные параметры системы.
        coefficients_matrix: Квадратная матрица коэффициентов (n x n).
        config: Конфигурация вычислений.

    Returns:
        BasicProbabilitySolver: Экземпляр аналитического решателя.

    Raises:
        ValueError: Если указан неподдерживаемый вычислительный движок.
    """
    match config.calculation_engine:
        case CalculationEngine.NUMPY:
            logger.info("Создан AnalyticalNumpyProbabilitySolver")
            return AnalyticalNumpyProbabilitySolver(
                transient_params,
                coefficients_matrix,
                config,
            )

        case CalculationEngine.MPMATH:
            logger.info("Создан AnalyticalMpmathProbabilitySolver")
            return AnalyticalMpmathProbabilitySolver(
                transient_params,
                coefficients_matrix,
                cast(MpmathComputationConfig, config),
            )

        case CalculationEngine.MERGED:
            logger.info("Создан AnalyticalMergedProbabilitySolver")
            return AnalyticalMergedProbabilitySolver(
                transient_params,
                coefficients_matrix,
                cast(MergedComputationConfig, config),
            )

        case _:
            logger.error(
                "Неподдерживаемый вычислительный движок для аналитического метода: "
                f"{config.calculation_engine}"
            )
            raise ValueError(
                f"Неподдерживаемый вычислительный движок: {config.calculation_engine}"
            )
