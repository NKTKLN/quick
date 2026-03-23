"""Фабричный модуль для создания решателей вероятностной модели СМО.

Предоставляет функцию `solvers_factory`, которая инстанцирует объект решателя
вероятностной системы на основе заданного метода расчета (`CalculationMethod`)
и вычислительного движка (`CalculationEngine`).
"""

from typing import cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.domain import CalculationEngine, CalculationMethod, ComputationConfig
from quick.domain.computation_config import (
    MergedComputationConfig,
    MpmathComputationConfig,
)
from quick.domain.models import TransientSystemParams
from quick.services.solvers.analytical import (
    AnalyticalMergedProbabilitySolver,
    AnalyticalMpmathProbabilitySolver,
    AnalyticalNumpyProbabilitySolver,
)
from quick.services.solvers.base import BasicProbabilitySolver
from quick.services.solvers.numerical import NumericalProbabilitySolver


def solvers_factory(
    params: TransientSystemParams,
    coefficients_matrix: NDArray[np.float64],
    config: ComputationConfig,
) -> BasicProbabilitySolver:
    """Создаёт и возвращает решатель вероятностной модели СМО.

    В зависимости от метода расчёта и вычислительного движка,
    инициализирует соответствующий решатель на основе переданных аргументов.

    Args:
        params (TransientSystemParams): Параметры СМО.
        coefficients_matrix (NDArray[np.float64]): Матрица коэффициентов системы
            уравнений размером (n x n).
        config (ComputationConfig): Конфигурация вычислений.

    Returns:
        BasicProbabilitySolver: Экземпляр соответствующего решателя.

    Raises:
        ValueError: Если передан неподдерживаемый метод расчёта или движок.
    """
    logger.debug(f"Вызван solvers_factory с параметрами: config={config}")

    if config.calculation_method == CalculationMethod.ANALYTICAL:
        match config.calculation_engine:
            case CalculationEngine.NUMPY:
                logger.info("Создан AnalyticalNumpyProbabilitySolver")
                return AnalyticalNumpyProbabilitySolver(
                    params, coefficients_matrix, config
                )
            case CalculationEngine.MPMATH:
                logger.info("Создан AnalyticalMpmathProbabilitySolver")
                return AnalyticalMpmathProbabilitySolver(
                    params, coefficients_matrix, cast(MpmathComputationConfig, config)
                )
            case CalculationEngine.MERGED:
                logger.info("Создан AnalyticalMergedProbabilitySolver")
                return AnalyticalMergedProbabilitySolver(
                    params, coefficients_matrix, cast(MergedComputationConfig, config)
                )
            case _:
                logger.error(
                    "Неподдерживаемый вычислительный движок: "
                    f"{config.calculation_engine}"
                )
                raise ValueError(
                    "Неподдерживаемый вычислительный движок: "
                    f"{config.calculation_engine}"
                )

    if config.calculation_method == CalculationMethod.NUMERICAL:
        logger.info("Создан NumericalProbabilitySolver")
        return NumericalProbabilitySolver(params, coefficients_matrix, config)

    raise ValueError(f"Неподдерживаемый метод расчета: {config.calculation_method}")
