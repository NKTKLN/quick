"""Модуль для моделирования систем массового обслуживания с Марковскими входными потоками в переходном режиме.

Содержит реализацию класса, выполняющего численное вычисление распределения
вероятностей состояний СМО во времени для систем с MAP-входным потоком.
"""

import numpy as np
from loguru import logger

from quick.services.matrix_builders import matrix_builder_factory
from quick.services.solvers import solvers_factory
from quick.services.systems.base import BaseMAPServerSystem


class TransientMAPServerStateSystem(BaseMAPServerSystem):
    """Модель СМО с MAP-входным потоком в переходном режиме.

    Класс предназначен для численного анализа системы массового обслуживания
    с марковски модулированным входным потоком (MAP) в нестационарной постановке.

    Используется для расчёта вероятностей состояний системы во времени
    на основе матрицы переходов и выбранного численного метода решения.
    """

    def calculate_probabilities(self) -> None:
        """Выполняет численный расчёт вероятностей состояний СМО.

        Raises:
            TypeError: Если self.params.transient_params не задан.
            TypeError: Если self.config не задан.
        """
        if self.params.transient_params is None:
            logger.error(
                "Невозможно выполнить расчёт вероятностей: transient_params не задан"
            )
            raise TypeError("transient_params должен быть не None")

        if self.config is None:
            logger.error("Невозможно выполнить расчёт вероятностей: config не задан")
            raise TypeError("config должен быть не None")

        logger.info("Начат расчёт вероятностей состояний СМО")
        try:
            self.log_average_lambda()
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            logger.warning(
                "Не удалось вычислить lambda_bar для диагностического вывода; "
                f"основной расчёт будет продолжен: {exc}"
            )
        transition_matrix = matrix_builder_factory(self.params).build()
        prob_solver = solvers_factory(
            params=self.params,
            coefficients_matrix=transition_matrix,
            config=self.config,
        )
        self._probabilities = prob_solver.calculate()
