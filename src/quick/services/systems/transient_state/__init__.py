"""Пакет классов систем массового обслуживания (СМО) в переходном режиме."""

from .map import TransientMAPServerStateSystem
from .multi import TransientMultiServerStateSystem

__all__ = ["TransientMultiServerStateSystem", "TransientMAPServerStateSystem"]
