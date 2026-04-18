# TODO

from dataclasses import dataclass, field

from .base_params import SystemParams
from .timeseries import TimeSeriesBaseSystemParams


@dataclass
class SimulationParams:
    """Конфигурация имитации.

    Attributes:
        trajectories: int: Количество траекторий Монте-Карло. (по умолчанию 10000).
        seed: Optional[int]: Начальное значение для ГПСЧ.
    """

    # TODO
    trajectories: int = 10_000
    seed: int | None = None

    def validate(self) -> None:
        """Проверяет корректность параметров."""
        if self.trajectories <= 0:
            raise ValueError("Количество траекторий должно быть положительным.")


@dataclass
class ImitationSystemParams(SystemParams):
    simulation_params: SimulationParams = field(default_factory=SimulationParams)
    time_series_params: TimeSeriesBaseSystemParams = field(
        default_factory=TimeSeriesBaseSystemParams
    )

    def validate(self) -> None:
        """Проверяет корректность параметров."""  # TODO
        super().validate()
        self.simulation_params.validate()
        self.time_series_params.validate()
