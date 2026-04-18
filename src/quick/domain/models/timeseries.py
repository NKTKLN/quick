# TODO

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class TimeSeries:
    """Временной ряд измерений.

    Описывает неубывающий массив временных точек и соответствующие им значения,
    используемые для моделирования динамических параметров системы.

    Attributes:
        times (NDArray[np.float64]): Массив времён измерений.
            Должен быть неубывающим и иметь такую же длину, как values.
        values (NDArray[np.float64]): Массив значений в соответствующие моменты времени.
            Должен иметь ту же форму, что и times.
    """

    times: NDArray[np.float64]
    values: NDArray[np.float64]

    def validate(self) -> None:
        """Проверяет корректность параметров."""
        if self.times.shape[0] != self.values.shape[0]:
            raise ValueError("times и values должны иметь одинаковую форму.")
        if self.times.shape[0] == 0:
            raise ValueError("Временной ряд пуст.")
        if not np.all(np.diff(self.times) >= 0):
            raise ValueError("times должен быть неубывающим.")

    def value_at(self, time_point: float) -> float:
        """Возвращает значение в момент времени time_point по схеме LOСF.

        Поведение:
        - Если time_point < times[0], возвращается values[0].
        - Если time_point > times[-1], возвращается values[-1].
        - Иначе возвращается последнее значение, соответствующее time <= time_point.

        Args:
            time_point (float): Момент времени, для которого требуется значение.

        Returns:
            float: Значение временного ряда, соответствующее моменту time_point.
        """
        idx = int(np.searchsorted(self.times, time_point, side="right") - 1)
        if idx < 0:
            idx = 0
        if idx >= self.values.shape[0]:
            idx = self.values.shape[0] - 1
        return self.values[idx]


@dataclass
class TimeSeriesBaseSystemParams:
    # TODO
    """Базовые параметры СМО.

    Attributes:
        mu_rate (NDArray[np.float64]): Интенсивность обслуживания заявок (μ > 0).
        nu_rate (NDArray[np.float64]): Интенсивность дополнительных процессов (ν > 0).
        lambda_rate (NDArray[np.float64]): Интенсивность поступления заявок (λ > 0).
    """

    mu_rate: TimeSeries | None = None
    nu_rate: TimeSeries | None = None
    lambda_rate: TimeSeries | None = None

    def validate(self) -> None:
        """Проверяет корректность параметров."""  # TODO
        if self.lambda_rate is not None:
            self.lambda_rate.validate()
        if self.mu_rate is not None:
            self.mu_rate.validate()
        if self.nu_rate is not None:
            self.nu_rate.validate()
