"""Модуль конфигурации логгирования приложения.

Содержит функцию setup_logger для настройки вывода логов в файл или консоль,
а также опцию полного отключения логирования для управления диагностической информацией.
"""

import sys

from loguru import logger

from app.settings import ConfigLoader


def setup_logger() -> None:
    """Инициализирует конфигурацию логгирования приложения.

    Настраивает логгер Loguru с параметрами из конфигурации: уровень логов,
    формат вывода, путь к файлу. При необходимости полностью отключает логгирование.
    """
    logger.remove()

    config = ConfigLoader.get_config()

    if config.disable_logging:
        return

    logger.add(
        sys.stdout,
        format=config.log_format,
        level=config.log_level,
        colorize=True,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    if config.log_path:
        logger.add(
            config.log_path,
            format=config.log_format,
            level=config.log_level,
            colorize=False,
            enqueue=True,
            backtrace=True,
            diagnose=True,
            rotation="10 MB",
            retention="10 days",
            compression="zip",
        )
