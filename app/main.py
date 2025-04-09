import logging
import numpy as np
from .logger import setup_logger
from .args_parser import parse_args
from .plot import plot_probabilities
from .system import ImpatientQueueSystem

# Получаем экземпляр логгера
logger = logging.getLogger(__name__)


def initialize_system(lam, muu, nuu, n):
    """Инициализация системы и расчет матриц."""
    impatient_queue_system = ImpatientQueueSystem(lam, muu, nuu, n)
    coefficient_matrix = impatient_queue_system.generate_coefficient_matrix()
    eigenvalues = impatient_queue_system.find_eigenvalues_of_coefficients_matrix(coefficient_matrix)
    p_values = impatient_queue_system.p_values_calculation(coefficient_matrix, eigenvalues)
    aa = impatient_queue_system.generate_aa_matrix(p_values)
    return impatient_queue_system, coefficient_matrix, eigenvalues, p_values, aa


def calculate_probabilities(impatient_queue_system, aa, p_values, eigenvalues, t, initial_conditions):
    """Расчет матриц M и P."""
    m = impatient_queue_system.generate_m_matrix(aa, p_values, t, eigenvalues, initial_conditions)
    p = impatient_queue_system.generate_p_matrix(m, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    return p


def main() -> None:
    args = parse_args()
    setup_logger(disable_logging=args.disable_logging, log_level=args.log_level, log_file=args.log_file)

    # Параметры системы
    lam = 89479
    muu = 134218.5
    nuu = 12435
    n = 30

    # Инициализация системы
    impatient_queue_system, _, eigenvalues, p_values, aa = initialize_system(lam, muu, nuu, n)

    # Временной интервал
    t = np.linspace(0, 0.0001, 1000)

    # Расчет вероятностей
    p = calculate_probabilities(impatient_queue_system, aa, p_values, eigenvalues, t, np.ones(30))

    # Построение графика
    plot_probabilities(p, t)


if __name__ == "__main__":
    main()
