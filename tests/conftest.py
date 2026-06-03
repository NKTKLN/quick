"""Общие фикстуры тестов программного комплекса QUICK."""

import numpy as np
import pytest
from loguru import logger

from quick.domain.params import TransientSystemParams


@pytest.fixture(autouse=True)
def disable_application_logging() -> None:
    """Отключает прикладное логирование во время выполнения тестов."""
    logger.disable("quick")


@pytest.fixture
def transient_params() -> TransientSystemParams:
    """Создаёт переходные параметры для тестов решателей.

    Returns:
        TransientSystemParams: Параметры переходного расчёта четырёх состояний.
    """
    times = np.linspace(0.0, 2.0, 9)
    initial = np.array([1.0, 0.0, 0.0, 0.0])
    return TransientSystemParams(times, np.zeros(4), initial)
