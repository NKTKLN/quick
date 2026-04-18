"""Фабричный модуль для создания решателей вероятностной модели СМО.

Предоставляет функцию `solvers_factory`, которая инстанцирует объект решателя
вероятностной системы на основе заданного метода расчета (`CalculationMethod`)
и вычислительного движка (`CalculationEngine`).
"""

from typing import cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.domain import (
    CalculationEngine,
    CalculationMethod,
    ComputationConfig,
    SystemType,
)
from quick.domain.computation_config import (
    MergedComputationConfig,
    MpmathComputationConfig,
)
from quick.domain.models import ImitationSystemParams, SystemParams

from .analytical import (
    AnalyticalMergedProbabilitySolver,
    AnalyticalMpmathProbabilitySolver,
    AnalyticalNumpyProbabilitySolver,
)
from .base import BasicProbabilitySolver
from .imitation import (
    MAPImitationProbabilitySolver,
    MultiServerImitationProbabilitySolver,
)
from .numerical import NumericalProbabilitySolver


def solvers_factory(
    params: type[SystemParams],
    coefficients_matrix: NDArray[np.float64],
    config: ComputationConfig,
) -> BasicProbabilitySolver:
    """Создаёт и возвращает решатель вероятностной модели СМО.

    В зависимости от метода расчёта и вычислительного движка,
    инициализирует соответствующий решатель на основе переданных аргументов.

    Args:
        params (SystemParams): Параметры СМО.
        coefficients_matrix (NDArray[np.float64]): Матрица коэффициентов системы
            уравнений размером (n x n).
        config (ComputationConfig): Конфигурация вычислений.

    Returns:
        BasicProbabilitySolver: Экземпляр соответствующего решателя.

    Raises:
        ValueError: Если передан неподдерживаемый метод расчёта или движок.
    """
    logger.debug(f"Вызван solvers_factory с параметрами: {config=}")

    if params.transient_params is None:
        logger.error("Переходные параметры не заданы.")
        raise ValueError("Переходные параметры должны быть заданы.")

    if config.calculation_method == CalculationMethod.ANALYTICAL:
        match config.calculation_engine:
            case CalculationEngine.NUMPY:
                logger.info("Создан AnalyticalNumpyProbabilitySolver")
                return AnalyticalNumpyProbabilitySolver(
                    params.transient_params, coefficients_matrix, config
                )
            case CalculationEngine.MPMATH:
                logger.info("Создан AnalyticalMpmathProbabilitySolver")
                return AnalyticalMpmathProbabilitySolver(
                    params.transient_params,
                    coefficients_matrix,
                    cast(MpmathComputationConfig, config),
                )
            case CalculationEngine.MERGED:
                logger.info("Создан AnalyticalMergedProbabilitySolver")
                return AnalyticalMergedProbabilitySolver(
                    params.transient_params,
                    coefficients_matrix,
                    cast(MergedComputationConfig, config),
                )
            case _:
                # TODO
                raise ValueError(
                    "Неподдерживаемый вычислительный движок: "
                    f"{config.calculation_engine}"
                )

    if config.calculation_method == CalculationMethod.NUMERICAL:
        logger.info("Создан NumericalProbabilitySolver")
        return NumericalProbabilitySolver(
            params.transient_params, coefficients_matrix, config
        )

    if config.calculation_method == CalculationMethod.IMITATION:
        if not isinstance(params, ImitationSystemParams):
            raise  # TODO

        match params.calculation_params.system_type:
            case SystemType.MULTI:
                return MultiServerImitationProbabilitySolver(params)
            case SystemType.MAP:
                return MAPImitationProbabilitySolver(params)
            case _:
                # TODO
                raise ValueError(
                    "Неподдерживаемый тип системы:"
                    f"{params.calculation_params.system_type}"
                )

    # TODO
    raise ValueError(f"Неподдерживаемый метод расчета: {config.calculation_method}")
