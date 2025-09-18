from abc import ABC, abstractmethod

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from app.domain.models.base import SystemParams


class BaseSystemBehavior(ABC):
    def __init__(self, params: SystemParams) -> None:
        """Инициализирует базовый построитель с параметрами модели.

        Args:
            params (BaseSystemParams): Параметры модели СМО.
        """
        self.params = params

        logger.debug(
            f"Инициализирован {self.__class__.__name__} с параметрами: "
            f"{params.base_params=}, {params.transient_params=}, {params.settings=}"
        )

    @property
    def lambda_rate(self) -> float:
        """Возвращает интенсивность поступления заявок.

        Raises:
            ValueError: Если параметры имеют тип MAPSystemParams.
        """
        return self.params.base_params.lambda_rate

    @abstractmethod
    def calculate_avg_system_length(
        self, probabilities: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Вычисляет среднее число заявок в системе.

        Returns:
            NDArray[float64]: Среднее число заявок в системе.
        """
        pass
