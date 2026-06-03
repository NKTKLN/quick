"""Модуль для многоканальных моделирования систем массового обслуживания в переходном режиме.

Содержит реализацию класса, выполняющего численное вычисление распределения
вероятностей состояний СМО во времени для многоканальных систем.
"""

from loguru import logger

from quick.services.matrix_builders import matrix_builder_factory
from quick.services.solvers import solvers_factory
from quick.services.systems.base import BaseMultiServerSystem


class TransientMultiServerStateSystem(BaseMultiServerSystem):
    """Модель многоканальной СМО в переходном режиме.

    Класс реализует численное вычисление вероятностей состояний многоканальной
    системы массового обслуживания с учетом переходной динамики.

    Используется для анализа временной эволюции системы по заданным параметрам
    входного потока, обслуживания и структуры пространства состояний.
    """

    def calculate_probabilities(self) -> None:
        """Выполняет численный расчёт вероятностей состояний СМО."""
        if self.params.transient_params is None:
            raise TypeError("transient_params должен быть не None")

        if self.config is None:
            raise TypeError("config должен быть не None")

        logger.info("Начат расчёт вероятностей состояний СМО")
        transition_matrix = matrix_builder_factory(self.params).build()
        prob_solver = solvers_factory(
            params=self.params,
            coefficients_matrix=transition_matrix,
            config=self.config,
        )
        self._probabilities = prob_solver.calculate()
