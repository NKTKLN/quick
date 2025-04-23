"""Модуль описывает систему массового обслуживания с нетерпеливыми заявками.

Содержит класс ImpatientQueueSystem, который реализует расчет вероятностных
характеристик СМО на основе заданных параметров. Расчёты производятся с
использованием матричных методов и собственных значений.
"""

import logging
from typing import List

import numpy as np
from numpy.typing import NDArray

from app.matrix_generators import CoefficientMatrixBuilder
from app.parameters import QueueSystemParameters, ThroughputQueueSystemParameters
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


class ThroughputQueueSystem:
    """Класс для расчета пропускной способности СМО с нетерпеливыми заявками.

    Осуществляет серийный расчет характеристик СМО для различных значений
    интенсивности ухода заявок (ν) с использованием матричного метода.
    """

    def __init__(self, params: ThroughputQueueSystemParameters) -> None:
        """Инициализирует систему с заданными параметрами.

        Args:
            params: Объект ThroughputQueueSystemParameters, содержащий:
                    - базовые параметры СМО
                    - диапазон значений интенсивности ухода заявок (ν)
        """
        self.params = params

    def calculate(self) -> List[NDArray[np.float64]]:
        """Выполняет расчет пропускной способности для различных значений ν.

        Процесс расчета включает:
        1. Итерацию по значениям интенсивности ухода заявок (ν)
        2. Расчет вероятностных характеристик для каждого ν
        3. Вычисление пропускной способности системы
        4. Формирование итогового массива результатов

        Returns:
            List[NDArray[np.float64]]: Список массивов пропускной способности
                                       для каждого значения ν. Каждый массив
                                       соответствует результатам расчета для
                                       конкретного значения интенсивности ухода.
        """
        throughput_results = []

        # Итерация по набору параметров с разными ν
        for single_param in self.params:
            # Создание и расчет СМО для текущего ν
            queue_system = ImpatientQueueSystem(single_param)
            probabilities = queue_system.calculate()

            # Расчет пропускной способности: (1 - p_loss) * lambda
            current_throughput = (1 - probabilities[-1]) * single_param.lambda_rate
            throughput_results.append(current_throughput)

            logger.info(
                "Рассчитана пропускная способность для ν=%.3f",
                single_param.nu_rate
            )

        return throughput_results
