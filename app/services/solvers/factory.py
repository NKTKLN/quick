"""Фабричный модуль для создания решателей вероятностной модели СМО.

Предоставляет функцию `solvers_factory`, которая инстанцирует объект решателя
вероятностной системы на основе заданного метода расчета (`CalculationMethod`)
и вычислительного движка (`CalculationEngine`).
"""

from typing import Any

from loguru import logger

from app.domain import CalculationEngine, CalculationMethod
from app.services.solvers.analytical import (
    AnalyticalMergedProbabilitySolver,
    AnalyticalMpmathProbabilitySolver,
    AnalyticalNumpyProbabilitySolver,
)
from app.services.solvers.base import BasicProbabilitySolver
from app.services.solvers.numerical import NumericalProbabilitySolver


def solvers_factory(
    calculation_method: CalculationMethod,
    calculation_engine: CalculationEngine | None,
    **kwargs: Any,
) -> BasicProbabilitySolver:
    """Создаёт и возвращает решатель вероятностной модели СМО.

    В зависимости от метода расчёта и вычислительного движка,
    инициализирует соответствующий решатель на основе переданных аргументов.

    Args:
        calculation_method (CalculationMethod): Метод вычисления (например, ANALYTICAL).
        calculation_engine (CalculationEngine | None): Тип вычислительного движка,
            поддерживаются: NUMPY, MPMATH, MERGED.
        **kwargs (Any): Дополнительные параметры, передаваемые в конструктор решателя
            (например: params, coefficients_matrix, config).

    Returns:
        BasicProbabilitySolver: Экземпляр соответствующего решателя.

    Raises:
        ValueError: Если передан неподдерживаемый метод расчёта или движок.
    """
    logger.debug(
        f"Вызван solvers_factory с параметрами: "
        f"calculation_method={calculation_method}, "
        f"calculation_engine={calculation_engine}"
    )

    if calculation_method == CalculationMethod.ANALYTICAL:
        match calculation_engine:
            case CalculationEngine.NUMPY:
                logger.info("Создан AnalyticalNumpyProbabilitySolver")
                return AnalyticalNumpyProbabilitySolver(**kwargs)
            case CalculationEngine.MPMATH:
                logger.info("Создан AnalyticalMpmathProbabilitySolver")
                return AnalyticalMpmathProbabilitySolver(**kwargs)
            case CalculationEngine.MERGED:
                logger.info("Создан AnalyticalMergedProbabilitySolver")
                return AnalyticalMergedProbabilitySolver(**kwargs)
            case _:
                logger.error(
                    f"Неподдерживаемый вычислительный движок: {calculation_engine}"
                )
                raise ValueError(
                    f"Неподдерживаемый вычислительный движок: {calculation_engine}"
                )

    if calculation_method == CalculationMethod.NUMERICAL:
        logger.info("Создан NumericalProbabilitySolver")
        return NumericalProbabilitySolver(**kwargs)

    raise ValueError(f"Неподдерживаемый метод расчета: {calculation_method}")
