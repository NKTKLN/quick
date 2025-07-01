"""Модуль базового построителя матриц коэффициентов для СМО с уходом заявок.

Определяет абстрактный класс `BaseMatrixBuilder`, предоставляющий интерфейс
и общее поведение для генерации матриц коэффициентов, используемых при решении
систем линейных дифференциальных уравнений, описывающих поведение различных
моделей массового обслуживания — одноканальных, многоканальных и с MAP-потоками.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from app.domain.models import MAPServerParams, MultiServerParams, SingleServerParams


class BaseMatrixBuilder(ABC):
    """Абстрактный базовый класс построителя матриц коэффициентов для СМО.

    Определяет интерфейс построения квадратной матрицы коэффициентов для систем
    дифференциальных уравнений, описывающих поведение СМО.
    """

    def __init__(
        self, params: SingleServerParams | MultiServerParams | MAPServerParams
    ) -> None:
        """Инициализирует базовый построитель с параметрами модели.

        Args:
            params (SingleServerParams | MultiServerParams | MAPServerParams):
                Параметры модели СМО.
        """
        self.params = params

    @abstractmethod
    def build(self) -> NDArray[np.float64]:
        """Абстрактный метод генерации матрицы коэффициентов для системы СМО.

        Returns:
            NDArray[np.float64]: Квадратная матрица коэффициентов.

        Raises:
            ValueError: Если параметры системы не являются нужным типом.
        """
