"""Модуль численного решателя вероятностей для однолинейной СМО.

Реализует численное решение системы дифференциальных уравнений Колмогорова
для модели одноканальной системы массового обслуживания (СМО) с уходом заявок.
Используется метод `solve_ivp` из библиотеки SciPy для получения распределения
вероятностей по времени.
"""

import logging
from typing import Callable, Optional, cast

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from app.domain import ComputationConfig
from app.domain.models import SingleServerParams
from app.services.solvers.base import BasicProbabilitySolver

# Инициализация логгера для текущего модуля
logger = logging.getLogger(__name__)


class NumericalProbabilitySolver(BasicProbabilitySolver):
    """Численный решатель вероятностей для одноканальной СМО.

    Вычисляет вероятности состояний с течением времени путём интегрирования
    системы дифференциальных уравнений Колмогорова с использованием
    метода Рунге-Кутты 4–5 порядка (RK45).
    """

    def __init__(
        self,
        params: SingleServerParams,
        coefficients_matrix: NDArray[np.float64],
        config: Optional[ComputationConfig] = None,
    ) -> None:
        """Инициализирует базовый решатель.

        Args:
            params (SingleServerParams): Параметры системы массового
                обслуживания.
            coefficients_matrix (NDArray[np.float64]): Матрица коэффициентов системы
                уравнений размером (n x n).
            config (Optional[ComputationConfig]): Конфигурация вычислений.
        """
        super().__init__(params, coefficients_matrix, config)

    def _transition_rates(
        self,
    ) -> Callable[[float, NDArray[np.float64]], NDArray[np.float64]]:
        """Возвращает функцию для расчёта производной вероятностей по времени.

        Returns:
            Callable: Функция, вычисляющая dP/dt = A * P для текущей
                матрицы коэффициентов.
        """

        def rates(t: float, P: NDArray[np.float64]) -> NDArray[np.float64]:
            """Вычисляет производную в момент времени t.

            Args:
                t (float): Текущий момент времени (не используется, т.к.
                    система однородна по времени).
                P (NDArray[np.float64]): Вектор вероятностей состояний в момент
                    времени t.

            Returns:
                NDArray[np.float64]: Производная вероятностей (dP/dt).
            """
            return self.coefficients_matrix @ P

        logger.debug("Создана функция расчёта переходных скоростей.")

        return rates

    def calculate(self) -> NDArray[np.float64]:
        """Выполняет численное интегрирование системы уравнений Колмогорова.

        Используется метод Runge-Kutta (RK45) для численного расчёта вероятностей
        состояний СМО на заданном временном интервале.

        Returns:
            NDArray[np.float64]: Матрица вероятностей состояний (размерность
                зависит от параметров СМО).
        """
        logger.info(
            "Начат расчёт вероятностей методом численного интегрирования (RK45)."
        )

        solution = solve_ivp(
            fun=self._transition_rates(),
            t_span=(self.params.time_array[0], self.params.time_array[-1]),
            y0=self.params.initial_probabilities,
            t_eval=self.params.time_array,
            method="RK45",
        )

        if not solution.success:
            logger.error(
                "Численный решатель не справился с задачей: %s", solution.message
            )
            raise RuntimeError("Численный решатель не справился с задачей.")

        logger.info("Численное интегрирование завершено успешно.")

        return cast(NDArray, solution.y)
