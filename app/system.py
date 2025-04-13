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
        if max_customers <= 1:
            raise ValueError("max_customers должен быть больше 1.")
        if any(rate <= 0 for rate in [lambda_rate, mu_rate, nu_rate]):
            raise ValueError("Все параметры интенсивности должны быть положительными.")

        self.lambda_rate = lambda_rate
        self.mu_rate = mu_rate
        self.nu_rate = nu_rate
        self.max_customers = max_customers 
        logger.info("Система инициализирована с λ=%.3f, μ=%.3f, ν=%.3f, n=%d", lambda_rate, mu_rate, nu_rate, max_customers) 
    
    def generate_coefficient_matrix(self) -> NDArray[np.float64]:
        """
        Генерация матрицы коэффициентов для системы уравнений.

        :return: матрица коэффициентов A для системы уравнений
        """
        coefficients_marix = np.zeros((self.max_customers, self.max_customers))

        # Заполним матрицу коэффициентами из системы уравнений
        for index in range(self.max_customers):
            if index == 0:
                # Первое уравнение: dp0/dt = -λp0 + μp1
                # На главной диагонали стоит -λ, а справа (в следующем столбце) стоит μ
                coefficients_marix[index, index] = -self.lambda_rate
                coefficients_marix[index, index+1] = self.mu_rate
            elif index == self.max_customers - 1:
                # Последнее уравнение: dpn+1/dt = λpn - (μ + nν)pn+1
                # Ниже главной диагонали стоит λ, а на главной диагонали -(μ + nν)
                coefficients_marix[index, index-1] = self.lambda_rate
                coefficients_marix[index, index] = -(self.mu_rate + (index - 1) * self.nu_rate)
            else:
                # Промежуточные уравнения: dpi/dt = λpi-1 - (μ + (i-1)ν + λ)pi + (μ + iν)pi+1
                coefficients_marix[index, index-1] = self.lambda_rate
                coefficients_marix[index, index] = -(self.mu_rate + (index - 1) * self.nu_rate + self.lambda_rate)
                coefficients_marix[index, index+1] = self.mu_rate + index * self.nu_rate
        
        logger.info("Матрица коэффициентов A для системы уравнений сгенерирована.")
        return coefficients_marix

    def _generate_L_matrix(self, coefficients_matrix: NDArray[np.float64], g: float) -> List[NDArray[np.float64]]:
        """
        Генерация матриц L для системы уравнений.

        :param coefficients_matrix: матрица коэффициентов A для системы уравнений
        :param g: параметр для модификации диагональных элементов
        :return: список матриц L
        """
        # Модификация диагональных элементов матрицы A
        modified_matrix = coefficients_matrix.copy()
        np.fill_diagonal(modified_matrix, modified_matrix.diagonal() - g)

        # Генерация матриц L
        l_matrices = []
        for index in range(self.max_customers):
            temp_matrix = modified_matrix.copy()
            temp_matrix[:, index] = -temp_matrix[:, 0]
            l_matrices.append(temp_matrix[:-1, 1:])
            logger.debug(f"Матрица L с индексом {index} сгенерирована.")
        
        logger.debug("Матрицы L для системы уравнений сгенерированы.")
        return l_matrices

    def p_values_calculation(self, coefficients_matrix: NDArray[np.float64], eigenvalues: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Вычисление значений P для каждого собственного значения g.

        :param coefficients_matrix: матрица коэффициентов A для системы уравнений
        :param eigenvalues: массив собственных значений матрицы A
        :return: словарь значений P
        """
        # Создаем массив для хранения значений P
        P_values = np.zeros((self.max_customers - 1, self.max_customers), dtype=np.float64)

        for g_value_index, g_value in enumerate(eigenvalues):
            logger.debug(f"Обработка собственного значения g[{g_value_index + 1}] = {g_value}.")
            L_matrices = self._generate_L_matrix(coefficients_matrix, g_value)

            # Вычисление значений P для матриц L
            L_det = linalg.det(L_matrices[0])
            P_values[:, g_value_index] = np.real(linalg.det(L_matrices[1:]) / L_det)
            logger.debug(f"Значения P для g[{g_value_index + 1}] = {g_value} вычислены.")

        logger.info("Вычисление значений P завершено.")
        return P_values

    def _calculate_a_p_values(self, p_values_matrix: NDArray[np.float64], shift: int = 0) -> NDArray[np.float64]:
        """
        Вычисление значений для A_P на основе матрицы значений P.

        :param p_values_matrix: матрица значений P
        :param shift: сдвиг для модификации матрицы
        :return: массив значений A_P
        """
        # Создание базовой матрицы xsi
        ones_row = np.ones(self.max_customers, dtype=np.float64)
        xsi_matrix = np.vstack([ones_row, p_values_matrix])

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

        logger.debug("Вычисление значений A_P завершено.")
        return a_p_values

    def generate_aa_matrix(self, p_values_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Генерация матрицы AA на основе матрицы значений P.

        :param p_values_matrix: матрица значений P
        :return: матрица AA
        """
        # Инициализация матрицы AA
        aa_matrix = np.empty((self.max_customers, self.max_customers), dtype=np.float64)
        
        # Заполнение матрицы AA
        for index in range(self.max_customers):
            a_p_values = self._calculate_a_p_values(p_values_matrix, index)
            aa_matrix[:, index] = a_p_values.T
            logger.debug(f"Столбец {index} для матрицы AA сгенерирован.")

        logger.info("Генерация матрицы AA завершена.")
        return aa_matrix

    def generate_m_matrix(self, aa_matrix: NDArray[np.float64], p_values_matrix: NDArray[np.float64], time_array: NDArray[np.float64], eigenvalues: NDArray[np.float64], initial_probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Генерация матрицы M на основе входных данных.

        :param aa_matrix: матрица AA
        :param p_values_matrix: матрица значений P
        :param time_array: массив времени
        :param eigenvalues: массив собственных значений
        :param initial_probabilities: массив начальных вероятностей [P1_1, P1_2, P1_3, P1_4, ...]
        :return: матрица M с дополнительным измерением времени
        """
        if initial_probabilities.shape[0] != p_values_matrix.shape[1]:
            raise ValueError("Размер initial_probabilities должен соответствовать p_values_matrix.shape[1].")
        
        # Создание матрицы p
        p_matrix = np.vstack([initial_probabilities, p_values_matrix])

        # Предварительное вычисление экспонент для каждого собственного значения
        exp_g_t = np.exp(np.outer(eigenvalues, time_array))

        # Инициализация матрицы M
        m_matrix = np.zeros((p_matrix.shape[0], aa_matrix.shape[1], len(time_array)), dtype=np.float64)

        # Заполнение матрицы M
        for i in range(p_matrix.shape[0]):
            for j in range(aa_matrix.shape[1]):
                m_matrix[i, j, :] = np.real((p_matrix[i, :] * aa_matrix[:, j]) @ exp_g_t)

        logger.info("Генерация матрицы M завершена.")
        return m_matrix
        
    def generate_p_matrix(self, m_matrix: NDArray[np.float64], initial_probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Вычисление матрицы вероятностей p на основе матрицы m и начальных вероятностей.

        :param m_matrix: матрица m
        :param initial_probabilities: массив начальных вероятностей [P1_0, P2_0, P3_0, P4_0, ...]
        :return: матрица p, где каждая строка соответствует вероятностям p1, p2, p3, p4, ...
        """
        if len(initial_probabilities) != m_matrix.shape[0]:
            raise ValueError("Размер initial_probabilities не соответствует размерности m_matrix.")

        # Вычисление матрицы p
        p_matrix = np.dot(initial_probabilities, m_matrix)

        logger.info("Вычисление матрицы p завершено.")
        return p_matrix
