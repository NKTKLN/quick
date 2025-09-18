import numpy as np
from loguru import logger
from numpy.typing import NDArray

from app.domain.models.base import SystemParams
from app.services.systems_behavior.base import BaseSystemBehavior


class MultiSystemBehavior(BaseSystemBehavior):
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
        N_b = np.sum(k_values[:, np.newaxis] * selected_probabilities, axis=0)

        logger.success("Среднее число заявок в системе вычислено")
        return N_b
