"""Модуль для анализа относительной пропускной способности СМО с нетерпеливыми заявками.

Включает классы для вычисления относительной пропускной способности одно- и
многолинейных СМО, в которых заявки могут покидать очередь при длительном ожидании.
Расчёты основаны на вероятностных моделях СМО с учётом интенсивностей поступления,
обслуживания и ухода заявок.
"""

from typing import cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from app.domain.models import MAPServerParams
from app.services.systems.base_throughput import BaseServerExpectedThroughputSystem


class ServerRelativeThroughputSystem(BaseServerExpectedThroughputSystem):
    """Класс для анализа относительной пропускной способности СМО.

    Реализует расчёты относительной пропускной способности с использованием
    вероятностной модели СМО.
    """

    def calculate(self) -> NDArray[np.float64] | list[NDArray[np.float64]]:
        """Выполняет полный расчёт относительной пропускной способности системы.

        Последовательность шагов:
            1. Расчёт вероятностей состояний системы
            2. Определение среднего количества ушедших заявок (n_b)
            3. Вычисление относительной пропускной способности: λ - ν * n_b
            4. Формирование массива итоговых значений

        Returns:
            NDArray[np.float64]: Значения относительной пропускной способности.

        Raises:
            ValueError: Если параметры системы являются экземпляром MAPServerParams.
        """
        logger.info("Начат расчёт относительной пропускной способности для СМО")

        if isinstance(self.params, MAPServerParams):
            logger.error("Получен недопустимый тип параметров: MAPServerParams")
            raise ValueError("Параметры не должны быть экземпляром MAPServerParams")

        probabilities = self._calculate_probabilities(self.params)
        N_b = self._calculate_expected_value(probabilities[-1], self.params.lambda_rate)

        relative_throughput = 1 - (self.params.nu_rate / self.params.lambda_rate) * N_b

        logger.success("Расчёт относительной пропускной способности завершён успешно")
        return cast(NDArray, relative_throughput)
