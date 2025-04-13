from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class QueueSystemParameters:
    lambda_rate: float                          # Интенсивность поступления заявок (λ)
    mu_rate: float                              # Интенсивность обслуживания заявок (μ)
    nu_rate: float                              # Интенсивность ухода нетерпеливых заявок (ν)
    max_customers: int                          # Максимальное количество заявок в системе (n)
    time_array: NDArray[np.float64]             # Массив времени
    state_variables: NDArray[np.float64]        # Массив переменных состояния
    initial_probabilities: NDArray[np.float64]  # Массив начальных вероятностей
