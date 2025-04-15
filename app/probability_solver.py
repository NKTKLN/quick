import logging
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .parameters import QueueSystemParameters

# Инициализация логгирования
logger = logging.getLogger(__name__)


class ProbabilitySolver:
    def __init__(
        self, 
        params: QueueSystemParameters, 
        coefficients_matrix: NDArray[np.float64], 
        eigenvalues: NDArray[np.float64]
    ) -> None:
        """
        Инициализация параметров системы.

        :param params: объект QueueSystemParameters с параметрами системы
        :param coefficients_matrix: матрица коэффициентов для системы уравнений
        :param eigenvalues: массив собственных значений матрицы коэффициентов
        """
        self.params = params
        self.coefficients_matrix = coefficients_matrix
        self.eigenvalues = eigenvalues

    def _generate_L_matrix(self, g: float) -> List[NDArray[np.float64]]:
        """
        Генерация матриц L для системы уравнений.

        :param g: параметр для модификации диагональных элементов
        :return: список матриц L
        """
        # Модификация диагональных элементов матрицы A
        modified_matrix = self.coefficients_matrix.copy()
        np.fill_diagonal(modified_matrix, modified_matrix.diagonal() - g)

        # Генерация матриц L
        l_matrices = []
        for index in range(modified_matrix.shape[1]):
            temp_matrix = modified_matrix.copy()
            temp_matrix[:, index] = -temp_matrix[:, 0]
            l_matrices.append(temp_matrix[:-1, 1:])
            logger.debug(f"Матрица L с индексом {index} сгенерирована.")
        
        logger.debug("Матрицы L для системы уравнений сгенерированы.")
        return l_matrices

    def p_values_calculation(self) -> NDArray[np.float64]:
        """
        Вычисление значений P для каждого собственного значения g.

        :return: словарь значений P
        """
        p_values = np.zeros((self.params.max_customers - 1, self.params.max_customers), dtype=np.float64)

        for index, g in enumerate(self.eigenvalues):
            logger.debug(f"Обработка собственного значения g[{index + 1}] = {g}.")
            L_matrices = self._generate_L_matrix(g)

            sign_L0, logdet_L0 = np.linalg.slogdet(L_matrices[0])
            if sign_L0 == 0:
                logger.warning(f"Определитель матрицы L[0] равен 0 при g[{index + 1}] = {g}, записываем NaN.")
                p_values[:, index] = np.nan
                continue

            for index2, l_matrix in enumerate(L_matrices[1:]):
                sign_Li, logdet_Li = np.linalg.slogdet(l_matrix)
                if sign_Li == 0:
                    p_values[index2, index] = np.nan
                    continue

                p_values[index2, index] = sign_Li * sign_L0 * np.exp(logdet_Li - logdet_L0)

            logger.debug(f"Значения P для g[{index + 1}] = {g} вычислены.")

        logger.info("Вычисление значений P завершено.")
        return p_values

    def _xsi_matrix_generator(
        self,
        p_values_matrix: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], Tuple[float, float]]:
        """
        Генерирует базовую матрицу xsi и вычисляет её определитель.

        :param p_values_matrix: матрица значений P
        :return: Кортеж, содержащий:
            - xsi_matrix: базовая матрица xsi
            - (sign, logdet): знак и логарифм определителя
        """
        ones_row = np.ones(p_values_matrix.shape[1], dtype=np.float64)
        xsi_matrix = np.vstack([ones_row, p_values_matrix])

        sign, logdet = np.linalg.slogdet(xsi_matrix)
        logger.debug(f"Определитель базовой матрицы xsi: sign = {sign}, logdet = {logdet}.")
        return xsi_matrix, (sign, logdet)

    def _calculate_a_p_values(
        self,
        xsi_matrix: NDArray[np.float64], 
        xsi_matrix_signature: Tuple[float, float],
        shift: int = 0
    ) -> NDArray[np.float64]:
        """
        Вычисление значений для A_P на основе матрицы значений P.

        :param xsi_matrix: базовя матрица xsi
        :param xsi_matrix_signature: кортеж (sign, logdet) базовой матрицы xsi
        :param shift: индекс строки, куда подставляется единица
        :return: массив значений A_P
        """
        sign_base, logdet_base = xsi_matrix_signature
        a_p_values = np.zeros(self.params.max_customers, dtype=np.float64)

        if sign_base == 0:
            logger.warning("Определитель базовой матрицы xsi равен 0. Все значения A_P будут NaN.")
            return a_p_values

        for index in range(self.params.max_customers):
            temp_xsi_matrix = xsi_matrix.copy()
            temp_xsi_matrix[:, index] = 0
            temp_xsi_matrix[shift, index] = 1

            sign_temp, logdet_temp = np.linalg.slogdet(temp_xsi_matrix)
            if sign_temp == 0:
                a_p_values[index] = np.nan
                logger.warning(f"Определитель temp_xsi_matrix для столбца {index} равен 0. A_P[{index}] = NaN.")
                continue

            a_p_values[index] = sign_temp * sign_base * np.exp(logdet_temp - logdet_base)
            logger.debug(f"Вычислено значение A_P[{index}] = {a_p_values[index]}.")

        logger.debug("Вычисление значений A_P завершено.")
        return a_p_values

    def generate_aa_matrix(self, p_values_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Генерация матрицы AA на основе матрицы значений P.

        :param p_values_matrix: матрица значений P
        :return: матрица AA
        """
        xsi_matrix, xsi_matrix_det = self._xsi_matrix_generator(p_values_matrix)

        aa_matrix = np.empty((self.params.max_customers, self.params.max_customers), dtype=np.float64)
        for index in range(aa_matrix.shape[1]):
            aa_matrix[:, index] = self._calculate_a_p_values(xsi_matrix, xsi_matrix_det, index).T
            logger.debug(f"Столбец {index} для матрицы AA сгенерирован.")

        logger.info("Генерация матрицы AA завершена.")
        return aa_matrix

    def generate_m_matrix(
        self,
        aa_matrix: NDArray[np.float64],
        p_values_matrix: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Генерация матрицы M на основе входных данных.

        :param aa_matrix: матрица AA
        :param p_values_matrix: матрица значений P
        :return: матрица M с дополнительным измерением времени
        """
        if self.params.state_variables.shape[0] != p_values_matrix.shape[1]:
            raise ValueError("Размер state_variables должен соответствовать p_values_matrix.shape[1].")
        
        p_matrix = np.vstack([self.params.state_variables, p_values_matrix])
        exp_g_t = np.exp(np.outer(self.eigenvalues, self.params.time_array))

        m_matrix = np.zeros((p_matrix.shape[0], aa_matrix.shape[1], len(self.params.time_array)), dtype=np.float64)
        for i in range(p_matrix.shape[0]):
            for j in range(aa_matrix.shape[1]):
                m_matrix[i, j, :] = np.real((p_matrix[i, :] * aa_matrix[:, j]) @ exp_g_t)
                logger.debug(f"Вычислено значение m_matrix[{i}, {j}, :].")

        logger.info("Генерация матрицы M завершена.")
        return m_matrix
        
    def generate_p_matrix(self, m_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Вычисление матрицы вероятностей p на основе матрицы m и начальных вероятностей.

        :param m_matrix: матрица m
        :return: матрица вероятностей P
        """
        if self.params.initial_probabilities.shape[0] != m_matrix.shape[0]:
            raise ValueError("Размер initial_probabilities не соответствует размерности m_matrix.")

        p_matrix = np.dot(self.params.initial_probabilities, m_matrix)
        logger.info("Вычисление матрицы p завершено.")
        return p_matrix
