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
from app.impatient_queue_system import ThroughputQueueSystem
from app.logger import setup_logger
from app.matrix_generators import MultiQueueSystemMatrixBuilder
from app.parameters import MultiQueueSystemParameters, ThroughputQueueSystemParameters
from app.plot import plot_throughput

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

    d = MultiQueueSystemParameters(
        lambda_rate=8333,
        mu_rate=10833,
        nu_rate=12345,
        max_customers=4,
        processor_count=2,
        time_array=np.linspace(0, 0.001, 1000, dtype=np.float64),
        state_variables=np.ones(4),
        initial_probabilities=np.concatenate((np.ones(1), np.zeros(3))),
    )

    print(MultiQueueSystemMatrixBuilder(d).build())

    return

    # Параметры системы
    params = ThroughputQueueSystemParameters(
        lambda_rate=8333,
        mu_rate=10833,
        nu_rate=np.array([1000, 10833, 10e5]),
        max_customers=4,
        time_array=np.linspace(0, 0.001, 1000, dtype=np.float64),
        state_variables=np.ones(4),
        initial_probabilities=np.concatenate((np.ones(1), np.zeros(3))),
    )

    # Инициализация системы
    impatient_queue_system = ThroughputQueueSystem(params)

    # for single_param in params:
    #     # Создание и расчет СМО для текущего ν
    #     queue_system = ImpatientQueueSystem(single_param)
    #     probabilities = queue_system.calculate()
    #     plot_probabilities(probabilities, params.time_array, args.save_plot)

    # Расчет вероятностей
    p = impatient_queue_system.calculate()

    # # Построение графика
    plot_throughput(p, params.time_array, args.save_plot)


if __name__ == "__main__":
    main()
