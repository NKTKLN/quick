"""Модуль конфигурации логгирования приложения.

Содержит функцию setup_logger для настройки вывода логов в файл или консоль.
"""

import sys

from loguru import logger

from quick.settings import ConfigLoader


def setup_logger() -> None:
    """Инициализирует конфигурацию логгирования приложения.

    Настраивает логгер Loguru с параметрами из конфигурации: уровень логов, формат
    вывода, путь к файлу. При необходимости полностью отключает логгирование.
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
        logger.debug(f"Добавление логгера для файла: {config.log_path}")
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
    else:
        logger.debug("Путь к лог-файлу не указан, логирование в файл отключено")

    logger.info("Логгирование инициализировано")
    logger.info(f"Уровень логирования установлен на: {config.log_level}")
    if config.log_path:
        logger.info(f"Логи будут сохраняться в файл: {config.log_path}")
    else:
        logger.info("Лог-файл не указан, вывод только в консоль")
