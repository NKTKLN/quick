"""Пакет базовых классов систем массового обслуживания (СМО)."""

from .base import BaseServerSystem
from .map import BaseMAPServerSystem
from .multi import BaseMultiServerSystem

__all__ = ["BaseMAPServerSystem", "BaseMultiServerSystem", "BaseServerSystem"]
