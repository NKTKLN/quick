"""Модуль модели системы массового обслуживания (СМО) в переходном режиме.

Содержит класс `TransientServerStateSystem`, реализующий численное моделирование
одноканальных и многоканальных СМО с нетерпеливыми заявками в переходном режиме.

Обеспечивает вычисление распределения вероятностей состояний системы,
пропускной способности, загрузки каналов и ключевых показателей качества обслуживания.
"""

from loguru import logger

from app.services.matrix_builders import matrix_builder_factory
from app.services.solvers import solvers_factory
from app.services.systems.base import BaseServerSystem


class TransientServerStateSystem(BaseServerSystem):
    """Класс для численного моделирования и анализа СМО в переходном режиме.

    Поддерживает как одноканальные, так и многоканальные системы с нетерпеливыми
    заявками. Предоставляет методы для расчёта распределения вероятностей состояний,
    пропускной способности, загрузки каналов и других ключевых показателей
    качества обслуживания.
    """

    def calculate_probabilities(self) -> None:
        """Выполняет численный расчёт вероятностей состояний СМО."""
        logger.info("Начат расчёт вероятностей состояний СМО")
        transition_matrix = matrix_builder_factory(self.params).build()
        prob_solver = solvers_factory(
            params=self.params.transient_params,
            coefficients_matrix=transition_matrix,
            config=self.config,
        )
        self._probabilities = prob_solver.calculate()
