"""Пакет классов систем массового обслуживания (СМО) в стационарном режиме."""

from .map import MAPSteadyStateSystem
from .multi import MultiServerSteadyStateSystem

__all__ = ["MultiServerSteadyStateSystem", "MAPSteadyStateSystem"]
