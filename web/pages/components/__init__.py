"""UI-компоненты для визуализации и конфигурации модели."""

from .calculation_config import render_calculation_config, render_imitation_config
from .initial_conditions import (
    render_initial_conditions,
    render_initial_probabilities,
    render_state_variables,
)
from .plot import plot_metric, plot_probabilities
from .plot3d import plot_phase_portrait, plot_sensor_surface, plot_surface
from .system_parameters import (
    get_intensity_parameters,
    get_system_mode_type,
    get_time_series_parameters,
    intensity_parameters,
    map_intensity_matrix,
    map_intensity_parameters,
)
from .time_settings import render_time_settings

__all__ = [
    "render_calculation_config",
    "render_imitation_config",
    "plot_probabilities",
    "plot_metric",
    "plot_surface",
    "plot_sensor_surface",
    "plot_phase_portrait",
    "render_state_variables",
    "render_initial_probabilities",
    "render_initial_conditions",
    "render_time_settings",
    "get_system_mode_type",
    "intensity_parameters",
    "map_intensity_parameters",
    "get_intensity_parameters",
    "map_intensity_matrix",
    "get_time_series_parameters",
]
