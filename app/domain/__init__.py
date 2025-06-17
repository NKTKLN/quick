"""Пакет с параметрами моделей систем массового обслуживания (СМО).

Экспортирует основные компоненты:
    - BasicSingleServerParams, BasicMultiServerParams — базовые параметры без ν.
    - SingleServerParams,
        MultiServerParams — параметры СМО с учетом интенсивности ухода (ν).
    - SingleServerThroughputParams,
        MultiServerThroughputParams — параметры для анализа пропускной способности.
    - ParamsNuIterator — итератор по значениям ν.
    - CalculationType, ComputationConfig,
        MpmathComputationConfig, MergedComputationConfig — конфигурации вычислений.
    - ConfigInitParams — параметры инициализации конфигурации модели.
"""

from .base_params import BasicMultiServerParams, BasicSingleServerParams
from .config_params import ConfigInitParams
from .iterators import ParamsNuIterator
from .probability import MultiServerParams, SingleServerParams
from .settings import (
    CalculationType,
    ComputationConfig,
    MergedComputationConfig,
    MpmathComputationConfig,
)
from .throughput import MultiServerThroughputParams, SingleServerThroughputParams

__all__ = [
    "BasicSingleServerParams",
    "BasicMultiServerParams",
    "SingleServerParams",
    "MultiServerParams",
    "SingleServerThroughputParams",
    "MultiServerThroughputParams",
    "ParamsNuIterator",
    "CalculationType",
    "ComputationConfig",
    "MpmathComputationConfig",
    "MergedComputationConfig",
    "ConfigInitParams",
]
