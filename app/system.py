import logging
import numpy as np
from typing import List
from scipy import linalg
from numpy.typing import NDArray

# Инициализация логгирования
logger = logging.getLogger(__name__)

class ImpatientQueueSystem:
    def __init__(self, lambda_rate: float, mu_rate: float, nu_rate: float, max_customers: int):
        """
        Инициализация параметров системы.

        :param lambda_rate: интенсивность поступления заявок (λ)
        :param mu_rate: интенсивность обслуживания заявок (μ)
        :param nu_rate: интенсивность ухода нетерпеливых заявок (ν)
        :param max_customers: максимальное количество заявок в системе (n)
        """
        self.lambda_rate = lambda_rate
        self.mu_rate = mu_rate
        self.nu_rate = nu_rate
        self.max_customers = max_customers  
    
    def generate_coefficient_matrix(self) -> NDArray[np.float64]:
        """
        Генерация матрицы коэффициентов для системы уравнений.

        :return: матрица коэффициентов A для системы уравнений
        """
        eye_matrix = np.eye(self.max_customers, dtype=np.float64)

        # Генерация коэффициентов для P_{n-1}(t)
        p_n_minus_1_coefficients = np.roll(eye_matrix, shift=1, axis=0)
        p_n_minus_1_coefficients[0, :] = 0
        p_n_minus_1_coefficients *= self.lambda_rate
        logger.info("Генерация коэффициентов для P_{n-1}(t) прошла успешно.")

        # Генерация коэффициентов для P_{n+1}(t)
        diag_elements = np.arange(0, self.max_customers)
        diag_matrix = np.diag(diag_elements)
        nu_matrix = np.roll(diag_matrix, shift=1, axis=1)
        nu_matrix[:, 0] = 0
        nu_matrix *= self.nu_rate

        mu_matrix = np.roll(eye_matrix, shift=-1, axis=0)
        mu_matrix[:, 0] = 0
        mu_matrix *= self.mu_rate

        p_n_plus_1_coefficients = nu_matrix + mu_matrix
        logger.info("Генерация коэффициентов для P_{n+1}(t) прошла успешно.")

        # Генерация коэффициентов для P_n(t)
        diag_elements_2 = np.arange(0, self.max_customers - 1)
        diag_elements_2 = np.concatenate((np.zeros(1), diag_elements_2))
        nu_matrix_2 = np.diag(diag_elements_2)
        nu_matrix_2 *= self.nu_rate

        mu_matrix_2 = np.eye(self.max_customers, dtype=np.float64)
        mu_matrix_2[0, 0] = 0
        mu_matrix_2 *= self.mu_rate

        lambda_matrix = np.eye(self.max_customers, dtype=np.float64)
        lambda_matrix *= self.lambda_rate

        p_n_coefficients = (nu_matrix_2 + mu_matrix_2 + lambda_matrix) * -1
        p_n_coefficients[self.max_customers-1, self.max_customers-1] = -(self.mu_rate + 2 * self.nu_rate)
        logger.info("Генерация коэффициентов для P_n(t) прошла успешно.")

        # Сложение матриц для финального результата
        coefficients_marix = p_n_minus_1_coefficients + p_n_coefficients + p_n_plus_1_coefficients
        logger.info("Матрица коэффициентов A для системы уравнений сгенерирована.")
        return coefficients_marix

    def _find_eigenvalues_of_coefficients_matrix(self, coefficients_marix: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Нахождение собственных значений матрицы коэффициентов.

        :param coefficients_marix: матрица коэффициентов A для системы уравнений
        :return: собственные значения матрицы A
        """
        eigenvalues_of_coefficients_marix = linalg.eigvals(coefficients_marix)
        logger.info("Собственные значения для матрицы A найдены.")
        return eigenvalues_of_coefficients_marix

    def _generate_L_matrix(self, coefficients_matrix: NDArray[np.float64], g: float) -> List[NDArray[np.float64]]:
        """
        Генерация матриц L для системы уравнений.

        :param coefficients_matrix: матрица коэффициентов A для системы уравнений
        :param g: параметр для модификации диагональных элементов
        :return: список матриц L
        """
        # Модификация диагональных элементов матрицы A
        eye_matrix = np.eye(self.max_customers, dtype=np.float64) * g
        modified_matrix = coefficients_matrix - eye_matrix

        # Генерация матриц L
        l_matrices = []
        for index in range(4):
            temp_matrix = modified_matrix.copy()
            temp_matrix[:, 0] *= -1
            temp_matrix[:, index] = temp_matrix[:, 0]
            l_matrices.append(temp_matrix[:-1, 1:])
            logger.debug(f"Матрица L с индексом {index} сгенерирована.")
        
        logger.info("Матрицы L для системы уравнений сгенерированы.")
        return l_matrices

    def p_values_calculation(self, coefficients_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Вычисление значений P для каждого собственного значения g.

        :param coefficients_matrix: матрица коэффициентов A для системы уравнений
        :return: словарь значений P
        """
        eigenvalues = self._find_eigenvalues_of_coefficients_matrix(coefficients_matrix)
        logger.info("Начало вычисления значений P для каждого g.")

        # Создаем массив для хранения значений P
        P_values = np.zeros((self.max_customers - 1, self.max_customers), dtype=np.float64)

        for g_value_index, g_value in enumerate(eigenvalues):
            logger.debug(f"Обработка собственного значения g[{g_value_index + 1}] = {g_value}.")
            L_matrices = self._generate_L_matrix(coefficients_matrix, g_value)

            # Вычисление детерминанта первой матрицы L
            L_det = linalg.det(L_matrices[0])
            logger.debug(f"Детерминант L[0] для g[{g_value_index + 1}] = {L_det}.")

            # Вычисление значений P для оставшихся матриц L
            for matrix_index, L_matrix in enumerate(L_matrices[1:]):
                calc = linalg.det(L_matrix) / L_det
                P_values[matrix_index, g_value_index] = calc
                logger.debug(f"Вычислено значение P[{matrix_index + 2}, {g_value_index + 1}] = {calc}.")

        logger.info("Вычисление значений P завершено.")
        return P_values

    def _calculate_a_p_values(self, p_values_matrix: NDArray[np.float64], shift: int = 0) -> NDArray[np.float64]:
        """
        Вычисление значений для A_P на основе матрицы значений P.

        :param p_values_matrix: матрица значений P
        :param shift: сдвиг для модификации матрицы
        :return: массив значений A_P
        """
        logger.info("Начало вычисления значений A_P.")
        
        # Создание базовой матрицы xsi
        xsi_matrix = np.ones((self.max_customers, self.max_customers), dtype=np.float64)
        xsi_matrix[1:, :] = p_values_matrix
        xsi_matrix_det = linalg.det(xsi_matrix)
        logger.debug(f"Определитель базовой матрицы xsi: {xsi_matrix_det}.")

        # Вычисление значений A_P
        a_p_values = np.zeros(self.max_customers, dtype=np.float64)
        for index in range(self.max_customers):
            temp_xsi_matrix = xsi_matrix.copy()
            temp_xsi_matrix[:, index] = 0
            temp_xsi_matrix[shift, index] = 1
            calc = linalg.det(temp_xsi_matrix) / xsi_matrix_det
            a_p_values[index] = calc
            logger.debug(f"Вычислено значение A_P[{index}] = {calc}.")

        logger.info("Вычисление значений A_P завершено.")
        return a_p_values
