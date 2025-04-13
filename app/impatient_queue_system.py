import logging

import numpy as np
from numpy.typing import NDArray

from .parameters import QueueSystemParameters
from .matrix_generators import CoefficientMatrixBuilder
from .probability_solver import ProbabilitySolver

# Инициализация логгирования
logger = logging.getLogger(__name__)


class ImpatientQueueSystem:
    def __init__(self, params: QueueSystemParameters) -> None:
        """
        Инициализация параметров системы.

        :param params: объект QueueSystemParameters с параметрами системы
        """
        self.params = params

    def calculate(self) -> NDArray[np.float64]:
        """
        Расчет системы.

        :return: матрица вероятностей P
        """
        a_matrix = CoefficientMatrixBuilder(self.params).build()

        eigenvalues = np.linalg.eigvals(a_matrix)
        logger.debug("Собственные значения рассчитаны: %s", eigenvalues)

        prob_solver = ProbabilitySolver(self.params, a_matrix, eigenvalues)
        
        p_values = prob_solver.p_values_calculation()
        aa_matrix = prob_solver.generate_aa_matrix(p_values)
        m_matrix = prob_solver.generate_m_matrix(aa_matrix, p_values)
        return prob_solver.generate_p_matrix(m_matrix)
