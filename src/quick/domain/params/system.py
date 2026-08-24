"""Параметры систем массового обслуживания (СМО).

Содержит классы с параметрами для стационарного и переходного режимов
одноканальных и многоканальных СМО. Поддерживает валидацию параметров
в зависимости от режима расчёта и типа системы.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from quick.domain.enums import SystemMode, SystemType

from .base import BaseSystemParams
from .map import MAPSystemParams
from .multi import MultiSystemParams
from .stability import StabilityParams


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
        """Проверяет корректность параметров.

        Raises:
            ValueError: При некорректных параметрах.
        """
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
class CalculationParams:
    """Настройки расчёта СМО.

    Attributes:
        system_type (SystemType): Тип системы.
        system_mode (SystemMode): Режим системы.
        sensor_index (int | None): Номер MAP-фазы (датчика), по состояниям
            которой ведётся расчёт. ``None`` — агрегированный режим, в котором
            характеристики суммируются по всем датчикам. При заданном ``j``
            во всех формулах суммирование ``sum_i P(k, i, t)`` заменяется
            единственным слагаемым ``P(k, j, t)``.
    """

    system_type: SystemType
    system_mode: SystemMode
    sensor_index: int | None = None


@dataclass
class SystemParams:
    """Объединённые параметры СМО.

    Содержит базовые параметры системы и, опционально, параметры переходного режима.

    Attributes:
        base_params (BaseSystemParams): Базовые параметры.
        transient_params (TransientSystemParams | None): Параметры переходного режима.
            Обязательны, если расчёт выполняется в переходном режиме.
        calculation_settings (CalculationSettings): Настройки типа и режима системы.
        stability_params (StabilityParams): Параметры оценки устойчивости:
            критический уровень характеристики и границы областей
            устойчивости.
    """

    base_params: BaseSystemParams
    calculation_params: CalculationParams
    transient_params: TransientSystemParams | None = None
    stability_params: StabilityParams = field(default_factory=StabilityParams)

    def validate(self) -> None:
        """Проверяет корректность параметров.

        Raises:
            ValueError: При некорректных параметрах.
        """
        self.base_params.validate()
        self.stability_params.validate()
        self._validate_sensor_index()

        if self.calculation_params.system_mode == SystemMode.TRANSIENT:
            if self.transient_params is None:
                raise ValueError("transient_params обязательны в переходном режиме")

            self.transient_params.validate()

            if self.calculation_params.system_type == SystemType.MULTI and isinstance(
                self.base_params, MultiSystemParams
            ):
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
            if self.calculation_params.system_type == SystemType.MAP and isinstance(
                self.base_params, MAPSystemParams
            ):
                expected_size = (
                    self.base_params.max_customers * self.base_params.sensor_count
                )
                if (
                    self.transient_params.initial_probabilities.shape[0]
                    != expected_size
                ):
                    raise ValueError(
                        "Размер начальных вероятностей должен совпадать с максимальным "
                        "числом заявок в системе возведенных в квадрат."
                    )

    def _validate_sensor_index(self) -> None:
        """Проверяет корректность номера выбранного датчика.

        Индивидуальный режим определён только для MAP-систем: у остальных
        типов состояние не раскладывается по датчикам.

        Raises:
            ValueError: Если номер датчика задан для не-MAP системы или
                выходит за границы диапазона.
        """
        sensor_index = self.calculation_params.sensor_index

        if sensor_index is None:
            return

        if self.calculation_params.system_type != SystemType.MAP or not isinstance(
            self.base_params, MAPSystemParams
        ):
            raise ValueError(
                "Расчёт по отдельному датчику доступен только для MAP-систем."
            )

        if not 0 <= sensor_index < self.base_params.sensor_count:
            raise ValueError(
                "Номер датчика должен лежать в диапазоне "
                f"[0, {self.base_params.sensor_count - 1}]."
            )
