"""Главный модуль запуска моделирования СМО с нетерпеливыми заявками.

Сценарий выполняет:
- Парсинг аргументов командной строки
- Инициализацию параметров СМО
- Расчёт вероятностей состояний
- Построение графиков

Используется для демонстрации работы модели и её визуализации.
"""

import logging

import numpy as np

from app.args_parser import parse_args
from app.impatient_queue_system import ImpatientQueueSystem
from app.logger import setup_logger
from app.parameters import QueueSystemParameters
from app.plot import plot_probabilities

# Получаем экземпляр логгера
logger = logging.getLogger(__name__)


def main() -> None:
    """Основная функция для запуска моделирования СМО с нетерпеливыми заявками.

    Выполняет последовательно:
    1. Парсинг аргументов командной строки
    2. Настройку системы логирования
    3. Инициализацию параметров системы
    4. Расчет вероятностных характеристик
    5. Визуализацию результатов

    Workflow:
        1. Создает параметры системы с фиксированными значениями
        2. Инициализирует систему массового обслуживания
        3. Вычисляет матрицу вероятностей состояний
        4. Строит графики полученных вероятностей

    Note:
        Параметры системы заданы жестко в коде для демонстрационных целей.
        В реальном использовании могут быть заменены на конфигурируемые параметры.

    Example:
        Запуск из командной строки:
        $ python main.py --log-level DEBUG --log-file simulation.log
    """
    # Парсинг аргументов командной строки
    args = parse_args()
    setup_logger(
        disable_logging=args.disable_logging,
        log_level=args.log_level,
        log_file=args.log_file,
    )

    # Параметры системы
    params = QueueSystemParameters(
        lambda_rate=89479,
        mu_rate=134218.5,
        nu_rate=12435,
        max_customers=31,
        time_array=np.linspace(0, 0.0001, 1000, dtype=np.float64),
        state_variables=np.ones(31),
        initial_probabilities=np.concatenate((np.zeros(30), np.ones(1))),
    )

    # Инициализация системы
    impatient_queue_system = ImpatientQueueSystem(params)

    # Расчет вероятностей
    p = impatient_queue_system.calculate()

    # Построение графика
    plot_probabilities(p, params.time_array)


if __name__ == "__main__":
    main()
