"""Модуль численного решения стационарных вероятностей для СМО с MAP-потоком.

Содержит реализацию класса MAPSteadyStateSystem, который рассчитывает
стационарные вероятности состояний системы массового обслуживания с
марковским входным потоком методом решения системы линейных уравнений.
"""

from typing import cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.domain.params import MAPSystemParams
from quick.services.matrix_builders.map import MultiSensorMAPServerMatrixBuilder
from quick.services.systems.base.map import BaseMAPServerSystem


class MAPSteadyStateSystem(BaseMAPServerSystem):
    """Класс для расчёта стационарных вероятностей состояний СМО с MAP-потоком.

    Решает систему стационарных уравнений:

        A · p = 0,
        sum(p) = 1,

    где A — матрица коэффициентов, построенная для MAP-системы.
    """

    def _build_coefficients_matrix(self) -> NDArray[np.float64]:
        """Строит матрицу коэффициентов стационарной системы.

        Returns:
            NDArray[np.float64]: Матрица коэффициентов.

        Raises:
            TypeError: Если базовые параметры не являются MAPSystemParams.
        """
        if not isinstance(self.params.base_params, MAPSystemParams):
            logger.error("Базовые параметры не являются экземпляром MAPSystemParams")
            raise TypeError("Базовые параметры должны быть экземпляром MAPSystemParams")

        builder = MultiSensorMAPServerMatrixBuilder(
            self.params.base_params,
            debug_logs=True,
        )

        return builder.build()

    def _solve_stationary_system(
        self,
        coefficients_matrix: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Решает систему стационарных уравнений с условием нормировки.

        Args:
            coefficients_matrix (NDArray[np.float64]): Матрица коэффициентов.

        Returns:
            NDArray[np.float64]: Вектор стационарных вероятностей.

        Raises:
            ValueError: Если матрица коэффициентов неквадратная.
        """
        if coefficients_matrix.ndim != 2:
            raise ValueError("Матрица коэффициентов должна быть двумерной.")

        rows, cols = coefficients_matrix.shape
        if rows != cols:
            raise ValueError(
                "Матрица коэффициентов должна быть квадратной: "
                f"{coefficients_matrix.shape=}"
            )

        logger.debug("Формирование системы с условием нормировки.")

        matrix = coefficients_matrix.copy()
        rhs = np.zeros(rows, dtype=np.float64)

        matrix[-1, :] = 1.0
        rhs[-1] = 1.0

        try:
            probabilities = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            logger.warning(
                "Матрица вырождена или плохо обусловлена. "
                "Используется решение методом наименьших квадратов."
            )
            probabilities = np.linalg.lstsq(matrix, rhs, rcond=None)[0]

        probabilities[np.isclose(probabilities, 0.0, atol=1e-14)] = 0.0

        if np.any(probabilities < -1e-10):
            logger.warning(
                "В найденном стационарном распределении есть отрицательные значения: "
                f"{probabilities.min()=}"
            )

        probabilities = np.clip(probabilities, 0.0, None)

        total = np.sum(probabilities)
        if not np.isclose(total, 1.0, atol=1e-10):
            logger.warning(f"Сумма вероятностей = {total}, выполняется перенормировка")
            probabilities /= total

        return cast(NDArray[np.float64], probabilities)

    def calculate_probabilities(self) -> None:
        """Вычисляет стационарные вероятности состояний СМО с MAP-потоком."""
        logger.info("Начат расчёт стационарных вероятностей состояний MAP-СМО")

        coefficients_matrix = self._build_coefficients_matrix()
        probabilities = self._solve_stationary_system(coefficients_matrix)

        logger.success(
            "Полный расчёт стационарных вероятностей MAP-СМО завершён успешно"
        )

        self._probabilities = probabilities.reshape(-1, 1)
