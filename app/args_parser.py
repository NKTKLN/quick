"""Модуль для парсинга аргументов командной строки.

Предоставляет CLI-интерфейс для настройки логгирования
в модели СМО с нетерпеливыми заявками.
"""

import argparse


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки для системы расчета СМО.

    Создает парсер аргументов для конфигурации системы моделирования СМО (Системы
    Массового Обслуживания) с нетерпеливыми заявками в переходном режиме. Поддерживает
    настройки логирования работы системы.

    Returns:
        argparse.Namespace: Объект с распарсенными аргументами командной строки.

    Example:
        >>> args = parse_args()
        >>> if args.disable_logging:
        ...     print("Логирование отключено")
        Пример вызова из командной строки:
        python main.py --log-level DEBUG --log-file simulation.log
    """
    parser = argparse.ArgumentParser(
        description="Система расчета математической модели СМО с нетерпеливыми заявками\
                     в переходгом режиме"
    )
    parser.add_argument(
        "--disable-logging", action="store_true", help="Отключить логгирование"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Уровень логгирования",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Имя файла для записи логов (если не указан, логи выводятся в консоль)",
    )
    return parser.parse_args()
