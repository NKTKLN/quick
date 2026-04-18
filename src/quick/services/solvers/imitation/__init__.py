"""Пакет имитационного решателя вероятностей для СМО."""

from .map import MAPImitationProbabilitySolver
from .multi import MultiServerImitationProbabilitySolver

__all__ = ["MultiServerImitationProbabilitySolver", "MAPImitationProbabilitySolver"]
