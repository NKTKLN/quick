"""Модуль для численного решения системы вероятностей СМО с нетерпеливыми заявками.

Содержит класс ProbabilitySolver, реализующий полный цикл вычисления вероятностных
характеристик системы массового обслуживания в переходном режиме.

Основной функционал:
- Расчет значений P через модифицированные матрицы L
- Построение матриц AA и M для временного анализа
- Финальное вычисление матрицы вероятностей P(t)
- Поддержка комплексных собственных значений и матричных операций
"""

import logging
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from app.parameters import QueueSystemParameters

# Инициализация логгирования
logger = logging.getLogger(__name__)


class ProbabilitySolver:
    """Класс для решения системы вероятностей СМО с нетерпеливыми заявками.

    Осуществляет вычисление вероятностных характеристик системы через:
    1. Генерацию матриц L на основе собственных значений
    2. Расчет значений P для каждого состояния
    3. Построение матриц AA и M
    4. Финальное вычисление матрицы вероятностей P
    """

    def __init__(
        self,
        params: QueueSystemParameters,
        coefficients_matrix: NDArray[np.float64],
        eigenvalues: NDArray[np.float64],
    ) -> None:
        """Инициализирует решатель вероятностей.

        Args:
            params: Параметры системы массового обслуживания
            coefficients_matrix: Матрица коэффициентов системы уравнений
                                 размером (n x n)
            eigenvalues: Массив собственных значений матрицы коэффициентов размером (n,)
        """
        self.params = params
        self.coefficients_matrix = coefficients_matrix
        self.eigenvalues = eigenvalues

    def _generate_l_matrix(self, g: float) -> List[NDArray[np.float64]]:
        """Генерирует список модифицированных матриц L для собственного значения.

        Args:
            g: Собственное значение, используемое для модификации матрицы

        Returns:
            List[NDArray[np.float64]]: Список матриц L размером (n-1 x n-1)
        """
        modified_matrix = self.coefficients_matrix.copy()
        np.fill_diagonal(modified_matrix, modified_matrix.diagonal() - g)

        l_matrices = []
        for index in range(modified_matrix.shape[1]):
            temp_matrix = modified_matrix.copy()
            temp_matrix[:, index] = -temp_matrix[:, 0]
            l_matrices.append(temp_matrix[:-1, 1:])
            logger.debug(f"Матрица L с индексом {index} сгенерирована.")

        logger.debug("Матрицы L для системы уравнений сгенерированы.")
        return l_matrices

    def p_values_calculation(self) -> NDArray[np.float64]:
        """Вычисляет значения P для всех собственных значений.

        Returns:
            NDArray[np.float64]: Матрица значений P размером (n-1 x n)
        """
        matrix_size = self.eigenvalues.shape[0]
        p_values = np.zeros((matrix_size - 1, matrix_size), dtype=np.float64)

        for index, g in enumerate(self.eigenvalues):
            logger.debug(f"Обработка собственного значения g[{index + 1}] = {g}.")
            l_matrices = self._generate_l_matrix(g)

            sign_l0, logdet_l0 = np.linalg.slogdet(l_matrices[0])
            if sign_l0 == 0:
                logger.warning(
                    f"Определитель матрицы L[0] равен 0 при g[{index + 1}] = {g}"
                )
                p_values[:, index] = np.nan
                continue

            for index2, l_matrix in enumerate(l_matrices[1:]):
                sign_li, logdet_li = np.linalg.slogdet(l_matrix)
                if sign_li == 0:
                    p_values[index2, index] = np.nan
                    continue

                p_values[index2, index] = (
                    sign_li * sign_l0 * np.exp(logdet_li - logdet_l0)
                )

            logger.debug(f"Значения P для g[{index + 1}] = {g} вычислены.")

        logger.info("Вычисление значений P завершено.")
        return p_values

    def _xsi_matrix_generator(
        self, p_values_matrix: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], Tuple[float, float]]:
        """Генерирует базовую матрицу xsi и вычисляет её определитель.

        Args:
            p_values_matrix: Матрица значений P размером (n-1 x n)

        Returns:
            Tuple[NDArray, Tuple[float, float]]:
                - Матрица xsi размером (n x n)
                - Кортеж (знак определителя, логарифм определителя)
        """
        ones_row = np.ones(p_values_matrix.shape[1], dtype=np.float64)
        xsi_matrix = np.vstack([ones_row, p_values_matrix])

        sign, logdet = np.linalg.slogdet(xsi_matrix)
        logger.debug(
            f"Определитель базовой матрицы xsi: sign = {sign}, logdet = {logdet}."
        )
        return xsi_matrix, (sign, logdet)

    def _calculate_a_p_values(
        self,
        xsi_matrix: NDArray[np.float64],
        xsi_matrix_signature: Tuple[float, float],
        shift: int = 0,
    ) -> NDArray[np.float64]:
        """Вычисляет значения A_P для матрицы xsi.

        Args:
            xsi_matrix: Базовая матрица xsi
            xsi_matrix_signature: Кортеж (знак, логарифм определителя)
            shift: Индекс строки для подстановки единицы

        Returns:
            NDArray[np.float64]: Вектор значений A_P размером (n,)
        """
        matrix_size = xsi_matrix.shape[0]
        sign_base, logdet_base = xsi_matrix_signature
        a_p_values = np.zeros(matrix_size, dtype=np.float64)

        if sign_base == 0:
            logger.warning(
                "Определитель базовой матрицы xsi равен 0. Все значения A_P будут NaN."
            )
            return a_p_values

        for index in range(matrix_size):
            temp_xsi_matrix = xsi_matrix.copy()
            temp_xsi_matrix[:, index] = 0
            temp_xsi_matrix[shift, index] = 1

            sign_temp, logdet_temp = np.linalg.slogdet(temp_xsi_matrix)
            if sign_temp == 0:
                a_p_values[index] = np.nan
                logger.warning(
                    f"Определитель temp_xsi_matrix для столбца {index} равен 0."
                )
                continue

            a_p_values[index] = (
                sign_temp * sign_base * np.exp(logdet_temp - logdet_base)
            )
            logger.debug(f"Вычислено значение A_P[{index}] = {a_p_values[index]}.")

        logger.debug("Вычисление значений A_P завершено.")
        return a_p_values

    def generate_aa_matrix(
        self, p_values_matrix: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Генерирует матрицу AA на основе значений P.

        Args:
            p_values_matrix: Матрица значений P размером (n-1 x n)

        Returns:
            NDArray[np.float64]: Матрица AA размером (n x n)
        """
        xsi_matrix, xsi_matrix_det = self._xsi_matrix_generator(p_values_matrix)

        matrix_size = p_values_matrix.shape[1]
        aa_matrix = np.empty((matrix_size, matrix_size), dtype=np.float64)
        for index in range(aa_matrix.shape[1]):
            aa_matrix[:, index] = self._calculate_a_p_values(
                xsi_matrix, xsi_matrix_det, index
            ).T
            logger.debug(f"Столбец {index} для матрицы AA сгенерирован.")

        logger.info("Генерация матрицы AA завершена.")
        return aa_matrix

    def generate_m_matrix(
        self, aa_matrix: NDArray[np.float64], p_values_matrix: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Генерирует матрицу M, объединяя временные и пространственные характеристики.

        Args:
            aa_matrix: Матрица AA размером (n x n)
            p_values_matrix: Матрица значений P размером (n-1 x n)

        Returns:
            NDArray[np.float64]: 3D матрица M размером (n x n x t)

        Raises:
            ValueError: Если размерности входных матриц не согласованы
        """
        if self.params.state_variables.shape[0] != p_values_matrix.shape[1]:
            raise ValueError(
                "Размер state_variables должен совпадать с p_values_matrix.shape[1]."
            )

        p_matrix = np.vstack([self.params.state_variables, p_values_matrix])
        exp_g_t = np.exp(np.outer(self.eigenvalues, self.params.time_array))

        m_matrix = np.zeros(
            (p_matrix.shape[0], aa_matrix.shape[1], len(self.params.time_array)),
            dtype=np.float64,
        )
        for i in range(p_matrix.shape[0]):
            for j in range(aa_matrix.shape[1]):
                m_matrix[i, j, :] = np.real(
                    (p_matrix[i, :] * aa_matrix[:, j]) @ exp_g_t
                )
                logger.debug(f"Вычислено значение m_matrix[{i}, {j}, :].")

        logger.info("Генерация матрицы M завершена.")
        return m_matrix

    def generate_p_matrix(self, m_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Вычисляет финальную матрицу вероятностей P.

        Args:
            m_matrix: Матрица M размером (n x n x t)

        Returns:
            NDArray[np.float64]: Матрица вероятностей P размером (n x t)

        Raises:
            ValueError: Если размерности матриц не согласованы
        """
        if self.params.initial_probabilities.shape[0] != m_matrix.shape[0]:
            raise ValueError(
                "Размер initial_probabilities не соответствует размерности m_matrix."
            )

        p_matrix: NDArray[np.float64] = np.dot(
            self.params.initial_probabilities, m_matrix
        )
        logger.info("Вычисление матрицы p завершено.")
        return p_matrix
