"""Фабричный модуль для создания решателей вероятностной модели СМО.

Предоставляет функцию solvers_factory, которая инстанцирует объект решателя
вероятностной системы на основе заданного метода расчета (CalculationMethod)
и вычислительного движка (CalculationEngine).
"""

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.domain import (
    CalculationMethod,
    ComputationConfig,
)
from quick.domain.models import ImitationSystemParams, SystemParams

from .analytical import analytical_solvers_factory
from .base import BasicProbabilitySolver
from .imitation import imitation_solvers_factory
from .numerical import numerical_solvers_factory


def solvers_factory(
    params: SystemParams,
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

    match config.calculation_method:
        case CalculationMethod.ANALYTICAL:
            return analytical_solvers_factory(
                params.transient_params, coefficients_matrix, config
            )

        case CalculationMethod.NUMERICAL:
            return numerical_solvers_factory(
                params.transient_params, coefficients_matrix, config
            )

        case CalculationMethod.IMITATION:
            if not isinstance(params, ImitationSystemParams):
                logger.error(
                    "Некорректный тип параметров для имитационного метода: "
                    f"ожидался ImitationSystemParams, получен {type(params)}"
                )
                raise TypeError(
                    "Для имитационного метода params должен быть экземпляром "
                    "ImitationSystemParams"
                )

            return imitation_solvers_factory(params)
