"""Модуль для аналитического решения системы вероятностей СМО с нетерпеливыми заявками.

Содержит реализацию гибридного решателя, который сочетает высокую точность библиотеки
mpmath с производительностью NumPy для вычисления динамики вероятностных состояний
системы массового обслуживания (СМО) в переходном режиме.
"""

from typing import Any, cast

import mpmath as mp  # type: ignore[import-untyped]
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.db import duckdb_cache
from quick.domain import ComputationConfig, MergedComputationConfig
from quick.domain.params import TransientSystemParams
from quick.services.solvers.analytical.mpmath_solver import (
    AnalyticalMpmathProbabilitySolver,
)
from quick.services.solvers.analytical.numpy_solver import (
    AnalyticalNumpyProbabilitySolver,
)


class AnalyticalMergedProbabilitySolver(AnalyticalMpmathProbabilitySolver):
    """Гибридный решатель, сочетающий точность mpmath и производительность numpy."""

    def __init__(
        self,
        params: TransientSystemParams,
        coefficients_matrix: NDArray[np.float64],
        config: MergedComputationConfig,
    ) -> None:
        """Инициализирует решатель вероятностей с повышенной точностью.

        Args:
            params (TransientSystemParams): Параметры СМО.
            coefficients_matrix (NDArray[np.float64]): Матрица коэффициентов системы
                уравнений размером (n x n).
            config (MergedComputationConfig): Конфигурация вычислений.
        """
        super().__init__(params, coefficients_matrix, config)

    @duckdb_cache("config.tolerance")
    def _get_invalid_indices(
        self, data_matrix: NDArray[np.float64 | Any], check_sum: bool = True
    ) -> NDArray[np.int64]:
        """Определяет индексы временных точек с некорректными значениями.

        Args:
            data_matrix (NDArray[np.float64 | Any]): Трехмерный массив значений M(t).
            check_sum (bool): Флаг проверки суммы по графикам (по умолчанию True).

        Returns:
            NDArray[np.int64]: Массив индексов последних некорректных временных
                точек для каждой пары.

        Raises:
            TypeError: Если self.config не является экземпляром MergedComputationConfig.
            ValueError: Если data_matrix имеет некорректную размерность или форму.
        """
        logger.debug("Начата проверка на некорректные значения в матрице M(t)")

        if not isinstance(self.config, MergedComputationConfig):
            logger.error(
                "Некорректный тип конфигурации: "
                f"ожидался MergedComputationConfig, получен {type(self.config)}"
            )
            raise TypeError("Конфиг должн быть экземпляром MergedComputationConfig")

        matrix_size = data_matrix.shape[0]
        time_steps = data_matrix.shape[2]

        invalid_indices = np.full((matrix_size, matrix_size), -1, dtype=np.int64)

        # Проверка значений каждого элемента
        for i in range(matrix_size):
            for j in range(matrix_size):
                for t in range(time_steps):
                    value = data_matrix[i, j, t]
                    if -self.config.tolerance <= value <= 1 + self.config.tolerance:
                        continue
                    invalid_indices[i, j] = max(invalid_indices[i, j], t)

        if not check_sum:
            logger.debug(f"Найдены некорректные индексы: {invalid_indices}")
            return invalid_indices

        # Проверка суммы по графикам
        sum_over_graphs = data_matrix.sum(axis=0)
        for j in range(matrix_size):
            for t in range(time_steps):
                value_sum = sum_over_graphs[j, t]
                if 1 - self.config.tolerance <= value_sum <= 1 + self.config.tolerance:
                    continue
                invalid_indices[:, j] = np.maximum(invalid_indices[:, j], t)

        logger.debug(f"Найдены некорректные индексы: {invalid_indices}")
        return invalid_indices

    def generate_m_matrix(
        self, eigenvalues: Any, xsi_matrix: Any
    ) -> NDArray[np.float64]:
        """Генерирует матрицу M(t) с улучшенной точностью, комбинируя numpy и mpmath.

        Производит начальное вычисление с помощью numpy, затем корректирует
        некорректные значения с использованием mpmath.

        Args:
            eigenvalues (Any): Массив собственных значений матрицы коэффициентов
                размером (n,).
            xsi_matrix (Any): Матрица собственных векторов (n x n).

        Returns:
            NDArray[np.float64]: 3D массив M(t) с исправленными значениями.
        """
        logger.info("Генерация матрицы M(t): старт с вычислений на numpy")
        numpy_eigenvalues = np.array(
            [float(mp.re(x)) for x in eigenvalues], dtype=np.float64
        )
        numpy_xsi_matrix = np.array(
            [[float(mp.re(x)) for x in row] for row in xsi_matrix.tolist()],
            dtype=np.float64,
        )

        analytical_numpy_solver = AnalyticalNumpyProbabilitySolver(
            self.params, self.coefficients_matrix, cast(ComputationConfig, self.config)
        )
        numpy_m_matrix = analytical_numpy_solver.generate_m_matrix(
            numpy_eigenvalues, numpy_xsi_matrix
        )
        logger.debug("Матрица M(t) по numpy сгенерирована")

        numpy_invalid_indices = self._get_invalid_indices(numpy_m_matrix)

        if np.all(numpy_invalid_indices == -1):
            logger.info("Все значения корректны, использование numpy достаточно")
            return numpy_m_matrix

        logger.info("Обнаружены некорректные значения, запускается коррекция с mpmath")
        with mp.workdps(self._precision):
            xsi_matrix_inv = mp.inverse(xsi_matrix)
            exp_g_t = self._generate_exp_matrix(eigenvalues)

        # Первичная коррекция неподходящих точек
        logger.debug("Начало первичной коррекции значений с mpmath")
        mpmath_m_matrix = self._generate_m_matrix(
            xsi_matrix, xsi_matrix_inv, exp_g_t, numpy_invalid_indices
        )
        correction_mask = ~mpmath_m_matrix.astype(bool)
        mpmath_m_matrix[correction_mask] = numpy_m_matrix[correction_mask]
        logger.debug("Первичная коррекция завершена")

        # Вторичная коррекции для сумм точек и их значений
        logger.debug("Начало вторичной проверки и коррекции с суммами")
        mpmath_invalid_indices = self._get_invalid_indices(
            mpmath_m_matrix, check_sum=True
        )
        end_m_matrix = self._generate_m_matrix(
            xsi_matrix, xsi_matrix_inv, exp_g_t, mpmath_invalid_indices
        )
        correction_mask = ~end_m_matrix.astype(bool)
        end_m_matrix[correction_mask] = mpmath_m_matrix[correction_mask]

        logger.info("Коррекция матрицы M(t) с использованием mpmath завершена")
        return end_m_matrix
