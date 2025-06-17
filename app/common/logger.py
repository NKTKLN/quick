"""Модуль настройки логгирования для приложения.

Предоставляет функцию setup_logger для конфигурации вывода логов в файл или консоль,
а также возможность полного отключения логгирования. Используется для контроля
вывода диагностической информации во время исполнения.
"""

import logging

from app.settings import ConfigLoader


def setup_logger() -> None:
    """Настраивает систему логгирования для приложения.

    Конфигурирует базовые настройки логирования с возможностью вывода как в консоль,
    так и в файл. Поддерживает полное отключение логирования при необходимости.
    """
    config = ConfigLoader.get_config()

    if config.disable_logging:
        logging.disable(logging.CRITICAL)
        return

    logging.basicConfig(
        level=config.log_level,
        format=config.log_format,
        filename=config.log_path,
        filemode="a",
    )
