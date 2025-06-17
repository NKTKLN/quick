"""Пакет с утилитами приложения.

Экспортирует основные компоненты:
    setup_logger — функция конфигурации вывода логов.
    PickleSerializer — класс для серилизации и десериализации объектов.
"""

from .logger import setup_logger
from .serialization import PickleSerializer

__all__ = ["setup_logger", "PickleSerializer"]
