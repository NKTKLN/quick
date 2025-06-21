"""Пакет решателей вероятностей для систем массового обслуживания.

Содержит классы для вычисления вероятностей состояний СМО с использованием
разных методов.
"""

from .analytical import (
    BasicProbabilitySolver,
    MergedProbabilitySolver,
    MpmathProbabilitySolver,
    NumpyProbabilitySolver,
)

__all__ = [
    "BasicProbabilitySolver",
    "NumpyProbabilitySolver",
    "MpmathProbabilitySolver",
    "MergedProbabilitySolver",
]
