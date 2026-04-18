"""Пакет решателей вероятностей для СМО."""

from .base import BasicProbabilitySolver
from .factory import solvers_factory

__all__ = ["BasicProbabilitySolver", "solvers_factory"]
