"""Модуль параметров имитационного моделирования СМО.

Реализует классы SimulationParams и ImitationSystemParams, которые описывают
настройки имитационного моделирования системы массового обслуживания (СМО),
включая параметры Монте-Карло и параметры, зависящие от времени.
"""

from dataclasses import dataclass, field

from .base_params import SystemParams
from .timeseries import TimeSeriesBaseSystemParams


@dataclass
class SimulationParams:
    """Конфигурация имитации.

    Attributes:
        trajectories (int): Количество траекторий Монте-Карло.
        seed (Optional[int]): Начальное значение для ГПСЧ.
    """

    trajectories: int = 10_000
    seed: int | None = None

    def validate(self) -> None:
        """Проверяет корректность параметров.

        Raises:
            ValueError: При некорректных параметрах.
        """
        if self.trajectories <= 0:
            raise ValueError("Количество траекторий должно быть положительным.")


@dataclass
class ImitationSystemParams(SystemParams):
    """Конфигурация имитации.

    Attributes:
        simulation_params (SimulationParams): Параметры для имитационной модели.
        time_series_params (TimeSeriesBaseSystemParams): Параметры СМО, зависымые от времени.
    """

    simulation_params: SimulationParams = field(default_factory=SimulationParams)
    time_series_params: TimeSeriesBaseSystemParams = field(
        default_factory=TimeSeriesBaseSystemParams
    )

    def validate(self) -> None:
        """Проверяет корректность параметров.

        Raises:
            ValueError: При некорректных параметрах.
        """
        super().validate()
        self.simulation_params.validate()
        self.time_series_params.validate()
