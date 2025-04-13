import logging
import numpy as np
from .logger import setup_logger
from .args_parser import parse_args
from .plot import plot_probabilities
from .system import ImpatientQueueSystem

# Получаем экземпляр логгера
logger = logging.getLogger(__name__)


def main() -> None:
    args = parse_args()
    setup_logger(disable_logging=args.disable_logging, log_level=args.log_level, log_file=args.log_file)

    # Параметры системы
    lam = 89479
    muu = 134218.5
    nuu = 12435
    n = 31
    initial_probabilities = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    # Временной интервал
    time_array = np.linspace(0, 0.0001, 1000)

    # Инициализация системы
    impatient_queue_system = ImpatientQueueSystem(lam, muu, nuu, n)

    # Расчет вероятностей
    p = impatient_queue_system.calculate(time_array, np.ones(n), initial_probabilities)

    # Построение графика
    plot_probabilities(p, time_array)


if __name__ == "__main__":
    main()
