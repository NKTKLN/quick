"""Модуль для моделирования систем массового обслуживания в переходном режиме.

Содержит реализации классов, выполняющих численное вычисление распределения
вероятностей состояний СМО во времени для многоканальных систем и систем
с MAP-входным потоком.

Классы модуля используют матричное представление динамики системы и численные
методы решения соответствующей системы дифференциальных или разностных уравнений
для получения вероятностных характеристик в переходном режиме.
"""

from loguru import logger

from quick.services.matrix_builders import matrix_builder_factory
from quick.services.solvers import solvers_factory
from quick.services.systems.base import BaseMAPServerSystem, BaseMultiServerSystem


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
            params=self.params.transient_params,
            coefficients_matrix=transition_matrix,
            config=self.config,
        )
        self._probabilities = prob_solver.calculate()


class TransientMAPServerStateSystem(BaseMAPServerSystem):
    """Модель СМО с MAP-входным потоком в переходном режиме.

    Класс предназначен для численного анализа системы массового обслуживания
    с марковски модулированным входным потоком (MAP) в нестационарной постановке.

    Используется для расчёта вероятностей состояний системы во времени
    на основе матрицы переходов и выбранного численного метода решения.
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
            params=self.params.transient_params,
            coefficients_matrix=transition_matrix,
            config=self.config,
        )
        self._probabilities = prob_solver.calculate()
