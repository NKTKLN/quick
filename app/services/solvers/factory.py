"""Фабричный модуль для создания решателей вероятностной модели СМО.

Предоставляет функцию `solvers_factory`, которая инстанцирует объект решателя
вероятностной системы на основе заданного метода расчета (`CalculationMethod`)
и вычислительного движка (`CalculationEngine`).
"""

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from app.domain import CalculationEngine, CalculationMethod, ComputationConfig
from app.domain.models import TransientSystemParams
from app.services.solvers.analytical import (
    AnalyticalMergedProbabilitySolver,
    AnalyticalMpmathProbabilitySolver,
    AnalyticalNumpyProbabilitySolver,
)
from app.services.solvers.base import BasicProbabilitySolver
from app.services.solvers.numerical import NumericalProbabilitySolver


def solvers_factory(
    params: TransientSystemParams,
    coefficients_matrix: NDArray[np.float64],
    config: ComputationConfig | None = None,
) -> BasicProbabilitySolver:
    """Создаёт и возвращает решатель вероятностной модели СМО.

    В зависимости от метода расчёта и вычислительного движка,
    инициализирует соответствующий решатель на основе переданных аргументов.

    Args:
        params (TransientSystemParams): Параметры СМО.
        coefficients_matrix (NDArray[np.float64]): Матрица коэффициентов системы
            уравнений размером (n x n).
        config (ComputationConfig | None, optional): Конфигурация вычислений.

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
                    params, coefficients_matrix, config
                )
            case CalculationEngine.MERGED:
                logger.info("Создан AnalyticalMergedProbabilitySolver")
                return AnalyticalMergedProbabilitySolver(
                    params, coefficients_matrix, config
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
