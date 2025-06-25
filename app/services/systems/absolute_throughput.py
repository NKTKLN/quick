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

from app.domain import (
    ComputationConfig,
    MergedComputationConfig,
    MpmathComputationConfig,
)
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
        if isinstance(self.params, MAPServerParams):
            raise ValueError("Параметры не должны быть экземпляром MAPServerParams")

        probabilities = self._calculate_probabilities(self.params)
        n_b = self._calculate_expected_value(self.params.lambda_rate, probabilities[-1])

        absolute_throughput = self.params.lambda_rate - self.params.nu_rate * n_b

        logger.info(
            "Рассчитана абсолютная пропускная способность для "
            f"ν = {self.params.nu_rate:.3f}"
        )

        return cast(NDArray, absolute_throughput)


class MAPServerAbsoluteThroughputSystem(ServerAbsoluteThroughputSystem):
    """Класс для анализа абсолютной пропускной способности СМО с MAP-потоками.

    Реализует расчёты абсолютной пропускной способности для различных значений ν с
    использованием вероятностной модели СМО с MAP-потоками.
    """

    def __init__(
        self,
        params: MAPServerParams,
        config: ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    ) -> None:
        """Инициализирует анализатор с параметрами однолинейной системы.

        Args:
            params (MAPServerThroughputParams): Параметры СМО.
            config: Конфигурация вычислений.
        """
        super().__init__(params, config)

    def calculate(self) -> list[NDArray[np.float64]]:
        """Выполняет полный расчёт абсолютной пропускной способности системы.

        Последовательность шагов:
            1. Итерация по значениям интенсивности поступления заявок (λ)
            2. Расчёт вероятностей состояний системы
            3. Определение среднего количества ушедших заявок (n_b)
            4. Вычисление абсолютной пропускной способности: λ - ν * n_b
            5. Формирование массива итоговых значений

        Returns:
            NDArray[np.float64]: Значения абсолютной пропускной способности.

        Raises:
            ValueError: Если параметры системы являются экземпляром MAPServerParams.
        """
        if not isinstance(self.params, MAPServerParams):
            raise ValueError("Параметры должны быть экземпляром MAPServerParams")

        probabilities = self._calculate_probabilities(self.params)

        absolute_throughput_results = []

        for lambda_rate in self.params.lambda_rate:
            n_b = self._calculate_expected_value(lambda_rate, probabilities[-1])
            current_absolute_throughput = lambda_rate - self.params.nu_rate * n_b
            absolute_throughput_results.append(current_absolute_throughput)

            logger.info(
                "Рассчитана абсолютная пропускная способность для "
                f"ν = {self.params.nu_rate:.3f}"
            )

        return absolute_throughput_results
