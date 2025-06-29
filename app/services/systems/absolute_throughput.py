"""Модуль для анализа абсолютной пропускной способности СМО с нетерпеливыми заявками.

Включает классы для вычисления абсолютной пропускной способности одно- и многолинейных
СМО, в которых заявки могут покидать очередь при длительном ожидании.
Расчёты основаны на вероятностных моделях СМО с учётом интенсивностей поступления,
обслуживания и ухода заявок.
"""

from typing import cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from app.domain.models import MAPServerParams
from app.services.systems.base_throughput import BaseServerExpectedThroughputSystem


class ServerAbsoluteThroughputSystem(BaseServerExpectedThroughputSystem):
    """Класс для анализа абсолютной пропускной способности СМО.

    Реализует расчёты абсолютной пропускной способности с использованием
    вероятностной модели СМО.
    """

    def calculate(self) -> NDArray[np.float64] | list[NDArray[np.float64]]:
        """Выполняет полный расчёт абсолютной пропускной способности системы.

        Последовательность шагов:
            1. Расчёт вероятностей состояний системы
            2. Определение среднего количества ушедших заявок (n_b)
            3. Вычисление абсолютной пропускной способности: λ - ν * n_b
            4. Формирование массива итоговых значений

        Returns:
            NDArray[np.float64]: Значения абсолютной пропускной способности.

        Raises:
            ValueError: Если параметры системы являются экземпляром MAPServerParams.
        """
        logger.info("Начат расчёт абсолютной пропускной способности для СМО")

        if isinstance(self.params, MAPServerParams):
            logger.error("Получен недопустимый тип параметров: MAPServerParams")
            raise ValueError("Параметры не должны быть экземпляром MAPServerParams")

        probabilities = self._calculate_probabilities(self.params)
        N_b = self._calculate_expected_value(probabilities[-1])

        absolute_throughput = self.params.lambda_rate - self.params.nu_rate * N_b

        logger.success("Расчёт абсолютной пропускной способности завершён успешно")
        return cast(NDArray, absolute_throughput)
