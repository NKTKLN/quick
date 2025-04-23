"""Модуль для моделирования систем массового обслуживания с нетерпеливыми заявками.

Модуль содержит классы для расчета вероятностных характеристик однолинейных и
многолинейных СМО с использованием матричных методов.

Основные классы:
1. ImpatientQueueSystem - моделирование однолинейной СМО с нетерпеливыми заявками
2. MultiImpatientQueueSystem - моделирование многолинейной СМО с нетерпеливыми заявками

Основные функции:
- Построение матриц переходов между состояниями системы
- Решение систем уравнений для стационарных вероятностей
- Расчет собственных значений матриц системы
"""

import logging

import numpy as np
from numpy.typing import NDArray

from app.matrix_generators import QueueSystemMatrixBuilder
from app.parameters import MultiQueueSystemParameters, QueueSystemParameters
from app.probability_solver import ProbabilitySolver

# Настройка логирования для отслеживания работы системы
logger = logging.getLogger(__name__)


class ImpatientQueueSystem:
    """Класс для моделирования СМО с нетерпеливыми заявками.

    Осуществляет расчет вероятностных характеристик СМО с использованием матричного
    метода на основе заданных параметров системы.
    """

    def __init__(self, params: QueueSystemParameters) -> None:
        """Инициализирует систему массового обслуживания с заданными параметрами.

        Args:
            params: Объект QueueSystemParameters, содержащий параметры системы:
                    - lambda_rate: интенсивность входящего потока
                    - mu_rate: интенсивность обслуживания
                    - nu_rate: интенсивность ухода заявок из очереди
                    - channel_count: количество каналов обслуживания
                    - queue_capacity: максимальная длина очереди
        """
        self.params = params

    def calculate(self) -> NDArray[np.float64]:
        """Выполняет полный расчет вероятностей состояний системы.

        Процесс расчета включает:
        1. Построение матрицы переходов системы
        2. Вычисление собственных значений матрицы
        3. Решение системы уравнений для вероятностей
        4. Построение итоговой матрицы вероятностей состояний

        Returns:
            NDArray[np.float64]: Матрица вероятностей размерностью (N+1)x(M+1),
                               где N - емкость системы, M - число источников.
                               P[i,j] - вероятность состояния с i заявками в системе
                               и j занятыми источниками.
        """
        transition_matrix = self._build_transition_matrix()
        eigenvalues = self._compute_eigenvalues(transition_matrix)
        probability_matrix = self._solve_probability_system(
            transition_matrix, eigenvalues
        )
        return probability_matrix

    def _build_transition_matrix(self) -> NDArray[np.float64]:
        """Строит матрицу переходов системы массового обслуживания.

        Returns:
            NDArray[np.float64]: Матрица переходов между состояниями системы.
        """
        transition_matrix = QueueSystemMatrixBuilder(self.params).build()
        return transition_matrix

    def _compute_eigenvalues(
        self, transition_matrix: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Вычисляет собственные значения матрицы переходов.

        Args:
            transition_matrix: Матрица переходов между состояниями системы.

        Returns:
            NDArray[np.float64]: Массив собственных значений матрицы.

        Note:
            Логирует рассчитанные собственные значения для отладки.
        """
        eigenvalues = np.linalg.eigvals(transition_matrix)
        real_eigenvalues = eigenvalues.real.astype(np.float64)
        logger.debug("Рассчитаны собственные значения: %s", real_eigenvalues)
        return real_eigenvalues

    def _solve_probability_system(
        self, transition_matrix: NDArray[np.float64], eigenvalues: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Решает систему уравнений для нахождения стационарных вероятностей.

        Args:
            transition_matrix: Матрица переходов системы.
            eigenvalues: Собственные значения матрицы переходов.

        Returns:
            NDArray[np.float64]: Матрица стационарных вероятностей состояний системы.

        Note:
            Использует ProbabilitySolver для поэтапного расчета:
            1. Вычисление промежуточных вероятностей (p_values)
            2. Построение расширенной матрицы (aa_matrix)
            3. Генерацию матрицы коэффициентов (m_matrix)
            4. Финальный расчет матрицы вероятностей
        """
        prob_solver = ProbabilitySolver(self.params, transition_matrix, eigenvalues)
        intermediate_probs = prob_solver.calculate_intermediate_probabilities()
        extended_matrix = prob_solver.build_extended_matrix(intermediate_probs)
        coefficient_matrix = prob_solver.build_coefficient_matrix(
            extended_matrix, intermediate_probs
        )
        return prob_solver.build_probability_matrix(coefficient_matrix)


class MultiImpatientQueueSystem:
    """Класс для моделирования многолинейной СМО с нетерпеливыми заявками.

    Осуществляет расчет вероятностных характеристик многолинейной СМО с использованием
    матричного метода на основе заданных параметров системы.
    """

    def __init__(self, params: MultiQueueSystemParameters) -> None:
        """Инициализирует систему массового обслуживания с заданными параметрами.

        Args:
            params: Объект MultiQueueSystemParameters, содержащий параметры системы:
                    - lambda_rate: интенсивность входящего потока
                    - mu_rate: интенсивность обслуживания
                    - nu_rate: интенсивность ухода заявок из очереди
                    - channel_count: количество каналов обслуживания
                    - queue_capacity: максимальная длина очереди
                    - processor_count: количество обслуживающих приборов в системе
        """
        self.params = params

    def _build_transition_matrix(self) -> NDArray[np.float64]:
        """Строит матрицу переходов многолинейной системы массового обслуживания.

        Returns:
            NDArray[np.float64]: Матрица переходов между состояниями системы.
        """
        transition_matrix = MultiImpatientQueueSystem(self.params).build()
        return transition_matrix
