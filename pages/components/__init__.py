"""UI-компоненты для визуализации и конфигурации модели.

Содержит функции ввода параметров системы, настройки времени,
начальных условий и построения графиков.
"""

from .calculation_config import calculation_config
from .initial_conditions import render_initial_probabilities, render_state_variables
from .plot import plot_probabilities, plot_throughput
from .system_parameters import (
    intensity_parameters,
    map_intensity_matrix,
    map_intensity_parameters,
    system_capacity_inputs,
    throughput_intensity_parameters,
)
from .time_settings import time_settings

__all__ = [
    "calculation_config",
    "plot_probabilities",
    "plot_throughput",
    "render_state_variables",
    "render_initial_probabilities",
    "time_settings",
    "system_capacity_inputs",
    "intensity_parameters",
    "throughput_intensity_parameters",
    "map_intensity_parameters",
    "map_intensity_matrix",
]
