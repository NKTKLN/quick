"""Модуль для анализа пропускной способности СМО с нетерпеливыми заявками.

Включает классы для вычисления пропускной способности одно- и многолинейных СМО,
в которых заявки могут покидать очередь при длительном ожидании.
Используется интеграция с вероятностными моделями СМО.
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
from app.services.systems.base_throughput import BaseServerThroughputSystem


class ServerThroughputSystem(BaseServerThroughputSystem):
    """Класс для анализа пропускной способности СМО.

    Реализует расчёты пропускной способности с использованием вероятностной модели СМО.
    """

    def calculate(self) -> NDArray[np.float64] | list[NDArray[np.float64]]:
        """Выполняет полный расчёт пропускной способности системы.

        Последовательность шагов:
            1. Расчёт вероятностей состояний системы
            2. Вычисление пропускной способности: (1 - p_loss) * λ
            3. Формирование массива итоговых значений

        Returns:
            NDArray[np.float64]: Список значений пропускной способности.

        Raises:
            ValueError: Если параметры системы являются MAPServerParams.
        """
        logger.info("Начат расчёт пропускной способности СМО")

        if isinstance(self.params, MAPServerParams):
            logger.error("Получены параметры MAPServerParams для обычной системы")
            raise ValueError("Параметры не должны быть экземпляром MAPServerParams")

        probabilities = self._calculate_probabilities(self.params)
        throughput = (1 - probabilities[-1]) * self.params.lambda_rate

        logger.success("Расчёт пропускной способности завершён успешно")
        return cast(NDArray, throughput)


class MAPServerThroughputSystem(ServerThroughputSystem):
    """Класс анализа пропускной способности СМО с MAP-потоками.

    Реализует расчёты пропускной способности для различных значений ν с использованием
    вероятностной модели СМО с MAP-потоками.
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
        """Выполняет полный расчёт пропускной способности системы.

        Последовательность шагов:
            1. Итерация по значениям интенсивности поступления заявок (λ)
            2. Расчёт вероятностей состояний системы
            3. Вычисление пропускной способности: (1 - p_loss) * λ
            4. Формирование массива итоговых значений

        Returns:
            list[NDArray[np.float64]]: Список значений пропускной способности.

        Raises:
            ValueError: Если параметры системы не являются MAPServerParams.
        """
        logger.info("Начат расчёт пропускной способности для MAP-СМО")

        if not isinstance(self.params, MAPServerParams):
            logger.error("Ожидались параметры MAPServerParams, но получены другие")
            raise ValueError("Параметры должны быть экземпляром MAPServerParams")

        probabilities = self._calculate_probabilities(self.params)

        throughput_results = []

        for lambda_rate in self.params.lambda_rate:
            current_throughput = (1 - probabilities[-1]) * lambda_rate
            throughput_results.append(current_throughput)
            logger.debug(f"Итерация расчёта для λ = {lambda_rate:.4} завершена")

        logger.success("Расчёт пропускной способности завершён успешно")
        return throughput_results
