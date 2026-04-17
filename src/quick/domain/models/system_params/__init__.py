"""Пакет с базовыми параметрами моделей систем массового обслуживания (СМО)."""

from .base import BaseSystemParams
from .map import MAPSystemParams
from .multi import MultiSystemParams

__all__ = ["BaseSystemParams", "MAPSystemParams", "MultiSystemParams"]
