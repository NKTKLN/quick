"""Модуль генерации матриц коэффициентов для систем массового обслуживания.

Реализует построение матриц для СМО с нетерпеливыми заявками, включая однолинейные,
многолинейные системы и модели с MAP-потоками.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

from app.domain import MAPServerParams, MultiServerParams, SingleServerParams

# Инициализация логгирования
logger = logging.getLogger(__name__)


class MatrixBuilder(ABC):
    """Абстрактный базовый класс построителя матриц коэффициентов для СМО.

    Определяет интерфейс построения квадратной матрицы коэффициентов для систем
    дифференциальных уравнений, описывающих поведение СМО.
    """

    def __init__(
        self, params: SingleServerParams | MultiServerParams | MAPServerParams
    ) -> None:
        """Инициализирует базовый построитель с параметрами модели.

        Args:
            params (SingleServerParams | MultiServerParams | MAPServerParams):
                Параметры модели СМО.
        """
        self.params = params

    @abstractmethod
    def build(self) -> np.ndarray[np.float64]:
        """Абстрактный метод генерации матрицы коэффициентов для системы СМО.

        Returns:
            np.ndarray[np.float64]: Квадратная матрица коэффициентов.
        """
        pass


class SingleServerMatrixBuilder(MatrixBuilder):
    """Построитель матрицы коэффициентов для одноканальной СМО с уходами заявок.

    Генерирует трёхдиагональную матрицу с учётом интенсивностей поступления,
    обслуживания и ухода нетерпеливых клиентов.
    """

    def __init__(self, params: SingleServerParams) -> None:
        """Инициализирует построитель одноканальной СМО.

        Args:
            params (SingleServerParams): Параметры одноканальной СМО.
        """
        super().__init__(params)

    def build(self) -> np.ndarray[np.float64]:
        """Формирует матрицу коэффициентов для одноканальной СМО.

        Returns:
            np.ndarray[np.float64]: Квадратная матрица коэффициентов размера n x n.
        """
        n = self.params.max_customers
        λ, μ, ν = self.params.lambda_rate, self.params.mu_rate, self.params.nu_rate

        coefficients_matrix = np.zeros((n, n), dtype=np.float64)

        for index in range(n):
            if index == 0:
                # Начальное состояние
                coefficients_matrix[index, index] = -λ
                coefficients_matrix[index, index + 1] = μ
            elif index == n - 1:
                # Последнее состояние
                coefficients_matrix[index, index - 1] = λ
                coefficients_matrix[index, index] = -(μ + (index - 1) * ν)
            else:
                # Промежуточные состояния
                coefficients_matrix[index, index - 1] = λ
                coefficients_matrix[index, index] = -(μ + (index - 1) * ν + λ)
                coefficients_matrix[index, index + 1] = μ + index * ν

        logger.info("Матрица коэффициентов для одноканальной СМО сгенерирована.")
        return coefficients_matrix


class MultiServerMatrixBuilder(MatrixBuilder):
    """Построитель матрицы коэффициентов для многоканальной СМО с уходами заявок.

    Учитывает количество обслуживающих приборов, интенсивность обслуживания
    и ухода из очереди.
    """

    def __init__(self, params: MultiServerParams) -> None:
        """Инициализирует построитель многоканальной СМО.

        Args:
            params (MultiServerParams): Параметры многоканальной СМО.
        """
        super().__init__(params)

    def build(self) -> np.ndarray[np.float64]:
        """Формирует матрицу коэффициентов для многоканальной СМО.

        Returns:
            np.ndarray[np.float64]: Квадратная матрица коэффициентов
                размера (n+m+1) x (n+m+1).
        """
        if not isinstance(self.params, MultiServerParams):
            raise TypeError("Ожидались параметры типа MultiServerParams")

        n, m = self.params.max_customers, self.params.processor_count
        λ, μ, ν = self.params.lambda_rate, self.params.mu_rate, self.params.nu_rate

        matrix_size = n + m + 1
        coefficients_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float64)

        for index in range(matrix_size):
            if index == 0:
                # Начальное состояние
                coefficients_matrix[index, index] = -λ
                coefficients_matrix[index, index + 1] = μ
            elif index == matrix_size - 1:
                # Конечное состояние
                coefficients_matrix[index, index - 1] = λ
                coefficients_matrix[index, index] = -(m * μ + n * ν)
            else:
                # Промежуточные состояния
                coefficients_matrix[index, index - 1] = λ
                coefficients_matrix[index, index] = -(
                    λ + min(index, m) * μ + max(0, index - m) * ν
                )
                coefficients_matrix[index, index + 1] = (
                    min(index + 1, m) * μ + max(0, index + 1 - m) * ν
                )

        logger.info("Матрица коэффициентов для многоканальной СМО сгенерирована.")
        return coefficients_matrix


class MAPServerMatrixBuilder(MatrixBuilder):
    """Построитель матрицы коэффициентов для СМО с MAP-потоками и уходами заявок.

    Использует матрицы вероятностей переходов p_rate и q_rate для построения
    четырёхмерной матрицы, сворачиваемой в двухмерную.
    """

    def __init__(self, params: MAPServerParams) -> None:
        """Инициализирует построитель СМО с MAP-потоком.

        Args:
            params (MAPServerParams): Параметры модели СМО с MAP-потоком.
        """
        super().__init__(params)

    def _d_0_matrix_generator(self) -> np.ndarray[np.float64]:
        """Генерирует матрицу D₀ по MAP-параметрам.

        Returns:
            np.ndarray[np.float64]: Матрица D₀ для текущих параметров потока.
        """
        matrix = self.params.p_rate.copy()
        matrix *= self.params.lambda_rate[:, np.newaxis]
        np.fill_diagonal(matrix, -self.params.lambda_rate)
        return matrix

    def _d_1_matrix_generator(self) -> np.ndarray[np.float64]:
        """Генерирует матрицу D₁ по MAP-параметрам.

        Returns:
            np.ndarray[np.float64]: Матрица D₁ для текущих параметров потока.
        """
        matrix = self.params.q_rate.copy()
        matrix *= self.params.lambda_rate[:, np.newaxis]
        return matrix

    def build(self) -> np.ndarray[np.float64]:
        """Формирует матрицу коэффициентов для СМО с MAP-потоками.

        Returns:
            np.ndarray[np.float64]: Квадратная матрица коэффициентов (n² x n²).
        """
        n = self.params.max_customers
        μ, ν = self.params.mu_rate, self.params.nu_rate

        d_0_t = self._d_0_matrix_generator().T
        d_1_t = self._d_1_matrix_generator().T

        coefficients_matrix = np.zeros((n, n, n, n), dtype=np.float64)

        for index in range(n):
            if index == 0:
                # Начальное состояние
                coefficients_matrix[index, index] = d_0_t
                coefficients_matrix[index, index + 1] = μ * np.eye(n)
            elif index == n - 1:
                # Конечное состояние
                coefficients_matrix[index, index - 1] = d_1_t
                coefficients_matrix[index, index] = d_0_t + d_1_t - (μ + ν) * np.eye(n)
            else:
                # Промежуточные состояния
                coefficients_matrix[index, index - 1] = d_1_t
                coefficients_matrix[index, index] = d_0_t - (
                    μ + ν * (0 if index in (0, 1) else 1)
                ) * np.eye(n)
                coefficients_matrix[index, index + 1] = (μ + index * ν) * np.eye(n)

        # Преобразование 4D-матрицы в 2D представление
        coefficients_matrix = coefficients_matrix.transpose(0, 2, 1, 3).reshape(
            n * n, n * n
        )

        logger.info("Матрица коэффициентов для СМО с MAP-потоками сгенерирована.")
        return coefficients_matrix
