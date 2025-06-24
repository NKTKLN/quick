"""Пакет конфигурации приложения.

Экспортирует основные классы для работы с конфигурацией:
    AppConfig — класс конфигурации приложения.
    ConfigLoader — класс для потокобезопасной инициализации и получения конфигурации.
"""

from .config import AppConfig, ConfigLoader

__all__ = ["AppConfig", "ConfigLoader"]
