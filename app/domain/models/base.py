"""Параметры систем массового обслуживания (СМО).

Содержит классы с параметрами для стационарного и переходного режимов
одноканальных и многоканальных СМО. Поддерживает валидацию параметров
в зависимости от режима расчёта и типа системы.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.domain.enums import (
    CalculationMode,
    SystemMode,
    SystemType,
)


@dataclass
class BaseSystemParams:
    """Базовые параметры СМО.

    Attributes:
        mu_rate (float): Интенсивность обслуживания заявок (μ > 0).
        nu_rate (float): Интенсивность дополнительных процессов (ν > 0).
        lambda_rate (float | NDArray[np.float64]): Интенсивность поступления
            заявок (λ > 0). Может быть скаляром или массивом.
        max_customers (int): Максимальное количество заявок в системе (n > 0).
        processor_count (int): Количество обслуживающих каналов (m).
    """

    mu_rate: float
    nu_rate: float
    lambda_rate: float | NDArray[np.float64]
    max_customers: int
    processor_count: int

    def validate(self) -> None:
        """Проверяет корректность параметров."""
        if self.mu_rate <= 0:
            raise ValueError("Интенсивность μ должна быть положительна.")
        if self.nu_rate <= 0:
            raise ValueError("Интенсивность ν должна быть положительна.")
        if isinstance(self.lambda_rate, np.ndarray):
            if np.any(self.lambda_rate <= 0) or self.lambda_rate.shape[0] == 0:
                raise ValueError("Интенсивность λ должна быть положительна.")
        else:
            if self.lambda_rate <= 0:
                raise ValueError("Интенсивность λ должна быть положительна.")
        if self.max_customers <= 0:
            raise ValueError("max_customers должна быть положительна.")
        if self.processor_count <= 0:
            raise ValueError("Переменная processor_count должна быть положительна.")


@dataclass
class TransientSystemParams:
    """Параметры СМО в переходном режиме.

    Содержит параметры, описывающие динамику системы во времени, включая
    временной массив, переменные состояния и начальные вероятности состояний.

    Attributes:
        time_array (NDArray[np.float64]): Массив временных точек расчета.
        state_variables (NDArray[np.float64]): Переменные состояния системы.
        initial_probabilities (NDArray[np.float64]): Начальные вероятности состояний.
    """

    time_array: NDArray[np.float64]
    state_variables: NDArray[np.float64]
    initial_probabilities: NDArray[np.float64]

    def validate(self) -> None:
        """Проверяет корректность параметров."""
        if self.time_array is None or self.time_array.shape[0] == 0:
            raise ValueError("Временной массив не задан.")
        if np.any(self.state_variables < 0):
            raise ValueError("Переменные состояния не могут быть отрицательными.")
        if self.state_variables.shape != self.initial_probabilities.shape:
            raise ValueError(
                "Размер переменных состояния должен совпадать с размером "
                "массива вероятностей."
            )
        if not np.isclose(np.sum(self.initial_probabilities), 1.0):
            raise ValueError("Сумма начальных вероятностей должна быть равна 1.")
        if np.any(self.initial_probabilities < 0) or (
            np.any(self.initial_probabilities > 1)
        ):
            raise ValueError(
                "Начальные вероятности должны находиться в диапазоне [0, 1]."
            )


@dataclass
class CalculationSettings:
    """Настройки расчёта СМО.

    Attributes:
        calculation_mode (CalculationMode): Режим расчёта.
        system_type (SystemType): Тип системы.
        system_mode (SystemMode): Режим системы.
    """

    calculation_mode: CalculationMode
    system_type: SystemType
    system_mode: SystemMode


@dataclass
class SystemParams:
    """Объединённые параметры СМО.

    Содержит базовые параметры системы и, опционально, параметры переходного режима.

    Attributes:
        base_params (BaseSystemParams): Базовые параметры.
        transient_params (TransientSystemParams | None): Параметры переходного режима.
            Обязательны, если расчёт выполняется в переходном режиме.
        settings (CalculationSettings): Настройки типа и режима системы.
    """

    base_params: BaseSystemParams
    settings: CalculationSettings
    transient_params: TransientSystemParams | None = None

    def validate(self) -> None:
        """Проверяет корректность параметров."""
        self.base_params.validate()

        if (
            self.settings.system_mode == SystemMode.TRANSIENT
            and self.transient_params is None
        ):
            raise

        if self.settings.system_mode == SystemMode.TRANSIENT:
            self.transient_params.validate()

            if self.settings.system_type == SystemType.MULTI:
                expected_size = (
                    self.base_params.max_customers
                    + self.base_params.processor_count
                    + 1
                )
                if (
                    self.transient_params.initial_probabilities.shape[0]
                    != expected_size
                ):
                    raise ValueError(
                        "Размер начальных вероятностей должен совпадать с максимальным "
                        "числом заявок в системе + колличество процессоров + 1."
                    )
            if self.settings.system_type == SystemType.MAP:
                expected_size = self.base_params.max_customers**2
                if (
                    self.transient_params.initial_probabilities.shape[0]
                    != expected_size
                ):
                    raise ValueError(
                        "Размер начальных вероятностей должен совпадать с максимальным "
                        "числом заявок в системе возведенных в квадрат."
                    )
