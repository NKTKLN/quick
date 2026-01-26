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
from app.domain.models.map import MAPSystemParams


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
            f"{params.max_customers=}, {params.processor_count=}, "
            f"{params.lambda_rate=}, {params.mu_rate=}, {params.nu_rate=}"
        )

    @abstractmethod
    def build(self) -> NDArray[np.float64]:
        """Абстрактный метод генерации матрицы коэффициентов для системы СМО.

        Returns:
            NDArray[np.float64]: Квадратная матрица коэффициентов.

        Raises:
            ValueError: Если параметры системы не являются нужным типом.
        """


class BaseMAPMatrixBuilder(BaseMatrixBuilder):
    """Базовый класс построителя матриц коэффициентов для СМО с MAP-потоками.

    Определяет интерфейс построения квадратной матрицы коэффициентов для систем
    дифференциальных уравнений, описывающих поведение СМО с MAP-потоками.
    """

    def __init__(self, params: MAPSystemParams) -> None:
        """Инициализирует базовый построитель с параметрами модели.

        Args:
            params (MAPSystemParams): Параметры модели СМО.
        """
        self.params = params

        logger.debug(
            f"Инициализирован MAPServerMatrixBuilder с параметрами: "
            f"max_customers={params.max_customers}, "
            f"processor_count={params.processor_count}, "
            f"lambda_rate={params.lambda_rate}, mu_rate={params.mu_rate}, "
            f"nu_rate={params.nu_rate}"
        )

    def _d_0_matrix_generator(self) -> NDArray[np.float64]:
        """Генерирует матрицу D₀ по MAP-параметрам.

        Returns:
            NDArray[np.float64]: Матрица D₀ для текущих параметров потока.

        Raises:
            ValueError: Если базовые параметры системы не являются MAPSystemParams.
        """
        if not isinstance(self.params, MAPSystemParams):
            logger.error("Базовые параметры не являются экземпляром MAPSystemParams")
            raise ValueError(
                "Базовые параметры должны быть экземпляром MAPSystemParams"
            )

        if self.params.p_rate.shape != self.params.q_rate.shape:
            logger.error("Матрицы p_rate и q_rate должны иметь одинаковую размерность.")
            raise ValueError(
                "Матрицы p_rate и q_rate должны иметь одинаковую размерность."
            )

        logger.debug("Генерация матрицы D₀ из p_rate и λ.")
        matrix = self.params.p_rate.copy()
        matrix *= self.params.lambda_rate[:, np.newaxis]
        np.fill_diagonal(matrix, -self.params.lambda_rate)
        logger.debug(f"Матрица D₀ сгенерирована. shape={matrix.shape}")
        return matrix

    def _d_1_matrix_generator(self) -> NDArray[np.float64]:
        """Генерирует матрицу D₁ по MAP-параметрам.

        Returns:
            NDArray[np.float64]: Матрица D₁ для текущих параметров потока.

        Raises:
            ValueError: Если базовые параметры системы не являются MAPSystemParams.
        """
        if not isinstance(self.params, MAPSystemParams):
            logger.error("Базовые параметры не являются экземпляром MAPSystemParams")
            raise ValueError(
                "Базовые параметры должны быть экземпляром MAPSystemParams"
            )

        logger.debug("Генерация матрицы D₁ из q_rate и λ.")
        matrix = self.params.q_rate.copy()
        matrix *= self.params.lambda_rate[:, np.newaxis]
        logger.debug(f"Матрица D₁ сгенерирована. shape={matrix.shape}")
        return matrix
