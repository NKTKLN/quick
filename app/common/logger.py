"""Модуль конфигурации логгирования приложения.

Содержит функцию setup_logger для настройки вывода логов в файл или консоль,
а также опцию полного отключения логирования для управления диагностической информацией.
"""

import logging

from app.settings import ConfigLoader


def setup_logger() -> None:
    """Инициализирует конфигурацию логгирования приложения.

    Настраивает базовый логгер с параметрами из конфигурации: уровень логов,
    формат вывода, путь к файлу. При необходимости полностью отключает логгирование.
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
