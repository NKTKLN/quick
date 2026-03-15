"""Модуль численного решателя вероятностей для однолинейной СМО.

Реализует численное решение системы дифференциальных уравнений Колмогорова
для модели одноканальной системы массового обслуживания (СМО) с уходом заявок.
Используется метод `solve_ivp` из библиотеки SciPy для получения распределения
вероятностей по времени.
"""

from collections.abc import Callable
from typing import cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from app.services.solvers.base import BasicProbabilitySolver


class NumericalProbabilitySolver(BasicProbabilitySolver):
    """Численный решатель вероятностей для одноканальной СМО.

    Вычисляет вероятности состояний с течением времени путём интегрирования системы
    дифференциальных уравнений Колмогорова с использованием метода Рунге-Кутты 4–5
    порядка (RK45).
    """

    def _transition_rates(
        self,
    ) -> Callable[[float, NDArray[np.float64]], NDArray[np.float64]]:
        """Возвращает функцию для расчёта производной вероятностей по времени.

        Returns:
            Callable: Функция, вычисляющая dP/dt = A * P для текущей
                матрицы коэффициентов.
        """

        def rates(_t: float, P: NDArray[np.float64]) -> NDArray[np.float64]:
            """Вычисляет производную в момент времени t.

            Args:
                _t (float): Текущий момент времени (не используется, т.к.
                    система однородна по времени).
                P (NDArray[np.float64]): Вектор вероятностей состояний в момент
                    времени t.

            Returns:
                NDArray[np.float64]: Производная вероятностей (dP/dt).
            """
            return self.coefficients_matrix @ P

        logger.debug("Создание функции расчёта переходных скоростей (dP/dt)")
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
        logger.debug(
            "Временной интервал: "
            f"{self.params.time_array[0]} — {self.params.time_array[-1]}"
        )
        logger.debug(f"Начальные вероятности: {self.params.initial_probabilities}")

        solution = solve_ivp(
            fun=self._transition_rates(),
            t_span=(self.params.time_array[0], self.params.time_array[-1]),
            y0=self.params.initial_probabilities,
            t_eval=self.params.time_array,
            method="RK45",
        )

        if not solution.success:
            logger.error(f"Решатель завершился с ошибкой: {solution.message}")
            raise RuntimeError("Численный решатель не справился с задачей.")

        logger.success("Численное интегрирование завершено успешно.")
        return cast(NDArray, solution.y)
