"""Пакет численного решателя вероятностей для СМО."""

from .factory import numerical_solvers_factory
from .solver import NumericalProbabilitySolver

__all__ = ["NumericalProbabilitySolver", "numerical_solvers_factory"]
