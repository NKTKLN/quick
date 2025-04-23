"""Модуль описывает систему массового обслуживания с нетерпеливыми заявками.

Содержит класс ImpatientQueueSystem, который реализует расчет вероятностных
характеристик СМО на основе заданных параметров. Расчёты производятся с
использованием матричных методов и собственных значений.
"""

import logging

import numpy as np
from numpy.typing import NDArray

from app.matrix_generators import CoefficientMatrixBuilder
from app.parameters import QueueSystemParameters
from app.probability_solver import ProbabilitySolver

# Инициализация логгирования
logger = logging.getLogger(__name__)


class ImpatientQueueSystem:
    """Класс для расчета системы массового обслуживания с нетерпеливыми заявками.

    Осуществляет расчет вероятностных характеристик СМО с использованием матричного
    метода на основе заданных параметров системы.
    """

    def __init__(self, params: QueueSystemParameters) -> None:
        """Инициализирует систему с заданными параметрами.

        Args:
            params: Объект QueueSystemParameters, содержащий все необходимые
                   параметры системы массового обслуживания.
        """
        self.params = params

    def calculate(self) -> NDArray[np.float64]:
        """Выполняет полный расчет системы массового обслуживания.

        Процесс расчета включает:
        1. Построение матрицы коэффициентов
        2. Вычисление собственных значений
        3. Расчет вероятностных характеристик
        4. Генерацию итоговой матрицы вероятностей

        Returns:
            NDArray[np.float64]: Матрица вероятностей P размерностью (N+1)x(M+1),
                                где N - максимальное число заявок в системе,
                                M - максимальное число источников заявок.
                                Элемент P[i,j] представляет вероятность нахождения
                                системы в состоянии i,j.
        """
        a_matrix = CoefficientMatrixBuilder(self.params).build()

        eigenvalues = np.linalg.eigvals(a_matrix)
        eigenvalues = eigenvalues.real.astype(np.float64)
        logger.debug("Собственные значения рассчитаны: %s", eigenvalues)

        prob_solver = ProbabilitySolver(self.params, a_matrix, eigenvalues)

        p_values = prob_solver.p_values_calculation()
        aa_matrix = prob_solver.generate_aa_matrix(p_values)
        m_matrix = prob_solver.generate_m_matrix(aa_matrix, p_values)
        return prob_solver.generate_p_matrix(m_matrix)
