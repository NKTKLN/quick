"""Модуль базовых параметров систем массового обслуживания.

Реализует класс BaseSystemParams, который задаёт основные параметры системы
массового обслуживания (СМО). Используется как базовый класс для более
сложных моделей, включая многоканальные системы и системы с марковскими
входными потоками.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class BaseSystemParams:
    """Базовые параметры СМО.

    Attributes:
        mu_rate (float): Интенсивность обслуживания заявок (μ > 0).
        nu_rate (float): Интенсивность ухода нетерпеливых заявок (ν >= 0).
            Значение ν = 0 допустимо и означает отсутствие ухода: заявка,
            попавшая в буфер, дожидается обслуживания. Этот случай нужен
            как нижняя граница развёртки по отношению ν/μ.
        lambda_rate (float | NDArray[np.float64]): Интенсивность поступления
            заявок (λ > 0). Может быть скаляром или массивом.
        max_customers (int): Максимальное количество заявок в системе (n > 0).
    """

    mu_rate: float
    nu_rate: float
    lambda_rate: float | NDArray[np.float64]
    max_customers: int

    def validate(self) -> None:
        """Проверяет корректность параметров.

        Raises:
            ValueError: При некорректных параметрах.
        """
        if self.mu_rate <= 0:
            raise ValueError("Интенсивность μ должна быть положительна.")
        if self.nu_rate < 0:
            raise ValueError("Интенсивность ν не может быть отрицательной.")
        if isinstance(self.lambda_rate, np.ndarray):
            if np.any(self.lambda_rate <= 0) or self.lambda_rate.shape[0] == 0:
                raise ValueError("Интенсивность λ должна быть положительна.")
        elif self.lambda_rate <= 0:
            raise ValueError("Интенсивность λ должна быть положительна.")
        if self.max_customers <= 0:
            raise ValueError("max_customers должна быть положительна.")
