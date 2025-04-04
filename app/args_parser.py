import argparse
from typing import Any


def parse_args() -> Any:
    """
    Парсинг аргументов командной строки.

    :return: Объект с аргументами.
    """
    parser = argparse.ArgumentParser(description="Система расчета математической модели СМО с нетерпеливыми заявками в переходгом режиме")
    parser.add_argument(
        '--disable-logging',
        action='store_true',
        help="Отключить логгирование"
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help="Уровень логгирования"
    )
    parser.add_argument(
        '--log-file',
        default=None,
        help="Имя файла для записи логов (если не указан, логи выводятся в консоль)"
    )
    return parser.parse_args()
