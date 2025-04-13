import logging
import numpy as np

from .logger import setup_logger
from .args_parser import parse_args
from .plot import plot_probabilities
from .parameters import QueueSystemParameters
from .impatient_queue_system import ImpatientQueueSystem

# Получаем экземпляр логгера
logger = logging.getLogger(__name__)


def main() -> None:
    args = parse_args()
    setup_logger(disable_logging=args.disable_logging, log_level=args.log_level, log_file=args.log_file)

    # Параметры системы
    params = QueueSystemParameters(
        lambda_rate=89479, 
        mu_rate=134218.5, 
        nu_rate=12435, 
        max_customers=31,
        time_array=np.linspace(0, 0.0001, 1000),
        state_variables=np.ones(31),
        initial_probabilities=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    )

    # Инициализация системы
    impatient_queue_system = ImpatientQueueSystem(params)

    # Расчет вероятностей
    p = impatient_queue_system.calculate()

    # Построение графика
    plot_probabilities(p, params.time_array)


if __name__ == "__main__":
    main()
