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

from quick.services.solvers.base import BasicProbabilitySolver


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
            f"{self.params.time_array[0]} - {self.params.time_array[-1]}"
        )
        logger.debug(f"Начальные вероятности: {self.params.initial_probabilities}")

        self._log_eigenvalues()

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

    def _log_eigenvalues(self) -> None:
        """Выводит в лог собственные значения матрицы коэффициентов.

        Численный метод RK45 не использует спектральное разложение для
        построения решения, однако спектр генераторной матрицы остаётся
        важной диагностической характеристикой переходного режима. Поэтому
        при запуске численного метода собственные значения вычисляются
        отдельно исключительно для вывода в лог и не влияют на результат
        интегрирования.
        """
        try:
            eigenvalues = np.linalg.eigvals(self.coefficients_matrix)
        except np.linalg.LinAlgError as exc:
            logger.warning(
                "Не удалось вычислить собственные значения матрицы для "
                f"диагностического вывода: {exc}"
            )
            return

        # Для генераторной матрицы удобно видеть сначала корни с наибольшей
        # действительной частью: стационарный (околонулевой) корень и затем
        # моды, определяющие скорость затухания переходного процесса.
        ordered = sorted(
            (complex(value) for value in eigenvalues),
            key=lambda value: (-value.real, abs(value.imag), value.imag),
        )

        logger.info(
            "Собственные значения матрицы коэффициентов при численном методе "
            f"(всего {len(ordered)}):"
        )
        for index, value in enumerate(ordered, start=1):
            if np.isclose(value.imag, 0.0, atol=1e-12):
                formatted = f"{value.real:.12g}"
            else:
                formatted = f"{value.real:.12g}{value.imag:+.12g}j"
            logger.info(f"  eig[{index:03d}] = {formatted}")

        max_real = max(value.real for value in ordered)

        # В формуле времени переходного режима из статьи используется не
        # модуль собственного значения, а минимальный ненулевой модуль его
        # действительной части: alpha_min = min |Re(gamma_i)|. Нулевой корень
        # генератора исключаем с запасом относительно ошибок eig для матриц Q
        # с интенсивностями порядка сотен/тысяч пакетов в секунду.
        matrix_scale = max(
            1.0, float(np.linalg.norm(self.coefficients_matrix, ord=np.inf))
        )
        zero_tol = max(1e-10, 1e-12 * matrix_scale)
        decay_rates = [
            abs(value.real) for value in ordered if abs(value.real) > zero_tol
        ]
        alpha_min = min(decay_rates, default=0.0)
        tau_max = 1.0 / alpha_min if alpha_min > 0.0 else float("inf")

        logger.info(
            "Спектральная сводка по формуле времени переходного режима: "
            f"max Re(eig) = {max_real:.12g}, "
            f"alpha_min = min |Re(eig)| = {alpha_min:.12g}, "
            f"tau_max = 1/alpha_min = {tau_max:.12g} с"
        )
