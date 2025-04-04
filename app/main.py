import logging
from .logger import setup_logger
from .args_parser import parse_args
from .system import ImpatientQueueSystem

# Получаем экземпляр логгера
logger = logging.getLogger(__name__)


def main() -> None:
    args = parse_args()
    setup_logger(disable_logging=args.disable_logging, log_level=args.log_level, log_file=args.log_file)

    # ==================================

    lam = 89479
    muu = 134218.5
    nuu = 12435
    n = 4

    impatient_queue_system = ImpatientQueueSystem(lam, muu, nuu, n)
    coefficient_matrix = impatient_queue_system.generate_coefficient_matrix()
    print(coefficient_matrix)

    p_values = impatient_queue_system.p_values_calculation(coefficient_matrix)
    print(p_values)

    d = impatient_queue_system.generate_aa_matrix(p_values)
    print(d)


if __name__ == "__main__":
    main()
