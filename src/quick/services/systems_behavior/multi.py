"""Содержит реализацию поведения многоканальной системы массового обслуживания.

Модуль определяет класс MultiSystemBehavior, который расширяет базовый
интерфейс поведения СМО для случая многоканальной системы обслуживания.
Класс отвечает за вычисление среднего числа заявок в системе на основе
вектора вероятностей состояний и параметров модели.

Основная сущность:
    MultiSystemBehavior — класс поведения многоканальной СМО, реализующий
    вычисление характеристик системы для модели с несколькими обслуживающими
    приборами.
"""

from typing import cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quick.domain.models.base import SystemParams
from quick.services.systems_behavior.base import BaseSystemBehavior


class MultiSystemBehavior(BaseSystemBehavior):
    """Класс поведения многоканальной системы массового обслуживания.

    Класс реализует вычисление характеристик СМО для моделей с несколькими
    обслуживающими приборами. Использует вероятности состояний системы и
    параметры модели для вычисления среднего числа заявок в системе.

    Attributes:
        params (SystemParams): Параметры системы массового обслуживания.
    """

    def __init__(self, params: SystemParams) -> None:
        """Инициализирует базовый построитель с параметрами модели.

        Args:
            params (BaseSystemParams): Параметры модели СМО.
        """
        super().__init__(params)

    def calculate_avg_system_length(
        self, probabilities: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Вычисляет среднее число заявок в системе.

        Returns:
            NDArray[float64]: Среднее число заявок в системе.
        """
        logger.info("Вычисление среднего числа заявок в системе")

        n = self.params.base_params.max_customers
        m = self.params.base_params.processor_count

        k_values = np.arange(1, n + 1)
        indices = m + k_values

        selected_probabilities = probabilities[indices, :]
        N_b = cast(
            NDArray[np.float64],
            np.sum(k_values[:, np.newaxis] * selected_probabilities, axis=0),
        )

        logger.success("Среднее число заявок в системе вычислено")
        return N_b
