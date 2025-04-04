import logging
import numpy as np
from typing import Any, List
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

    def find_eigenvalues_of_coefficients_matrix(self, coefficients_marix: NDArray[np.float64]) -> Any:
        """
        Нахождение собственных значений матрицы коэффициентов.

        :param coefficients_marix: матрица коэффициентов A для системы уравнений
        :return: собственные значения матрицы A
        """
        eigenvalues_of_coefficients_marix = linalg.eigvals(coefficients_marix)
        logger.info("Собственные значения для матрицы A найдены.")
        return eigenvalues_of_coefficients_marix

    def generate_L_matrix(self, coefficients_matrix: NDArray[np.float64], g: float) -> List[NDArray[np.float64]]:
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
