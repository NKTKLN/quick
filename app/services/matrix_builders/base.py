"""Модуль базового построителя матриц коэффициентов для СМО с уходом заявок.

Определяет абстрактный класс `BaseMatrixBuilder`, предоставляющий интерфейс
и общее поведение для генерации матриц коэффициентов, используемых при решении
систем линейных дифференциальных уравнений, описывающих поведение различных
моделей массового обслуживания — одноканальных, многоканальных и с MAP-потоками.
"""

from abc import ABC, abstractmethod

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from app.domain.models import BaseSystemParams


class BaseMatrixBuilder(ABC):
    """Абстрактный базовый класс построителя матриц коэффициентов для СМО.

    Определяет интерфейс построения квадратной матрицы коэффициентов для систем
    дифференциальных уравнений, описывающих поведение СМО.
    """

    def __init__(self, params: BaseSystemParams) -> None:
        """Инициализирует базовый построитель с параметрами модели.

        Args:
            params (BaseSystemParams): Параметры модели СМО.
        """
        self.params = params

        logger.debug(
            f"Инициализирован {self.__class__.__name__} с параметрами: "
            f"max_customers={params.max_customers}, "
            f"processor_count={params.processor_count}, "
            f"lambda_rate={params.lambda_rate}, mu_rate={params.mu_rate}, "
            f"nu_rate={params.nu_rate}"
        )

    @abstractmethod
    def build(self) -> NDArray[np.float64]:
        """Абстрактный метод генерации матрицы коэффициентов для системы СМО.

        Returns:
            NDArray[np.float64]: Квадратная матрица коэффициентов.

        Raises:
            ValueError: Если параметры системы не являются нужным типом.
        """
