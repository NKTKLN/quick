"""Пакет решателей вероятностей для систем массового обслуживания."""

from .base import BasicProbabilitySolver
from .factory import solvers_factory

__all__ = ["BasicProbabilitySolver", "solvers_factory"]
