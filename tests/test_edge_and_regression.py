"""Тесты предельных случаев и регрессионных значений модели."""

import math

import numpy as np

from quick.domain import ComputationConfig
from quick.domain.params import MultiSystemParams, TransientSystemParams
from quick.services.matrix_builders.multi import MultiServerMatrixBuilder
from quick.services.solvers.analytical.numpy_solver import (
    AnalyticalNumpyProbabilitySolver,
)


def _solve_two_state(lam: float, mu: float, times: np.ndarray) -> np.ndarray:
    """Решает систему M/M/1/1 на заданной временной сетке."""
    base = MultiSystemParams(mu, 0.0, lam, max_customers=0, processor_count=1)
    matrix = MultiServerMatrixBuilder(base).build()
    transient = TransientSystemParams(times, np.zeros(2), np.array([1.0, 0.0]))
    return AnalyticalNumpyProbabilitySolver(
        transient, matrix, ComputationConfig(disable_cache=True)
    ).calculate()


def test_degenerate_no_abandonment_generator_conserves_probability() -> None:
    """Проверяет сохранение вероятности при отключённых уходах."""
    base = MultiSystemParams(2.0, 0.0, 3.0, max_customers=4, processor_count=2)
    matrix = MultiServerMatrixBuilder(base).build()
    np.testing.assert_allclose(matrix.sum(axis=0), 0.0, atol=1e-15)
    assert np.all(np.diag(matrix) < 0)


def test_capacity_zero_reduces_to_closed_form_loss_system() -> None:
    """Сверяет решение M/M/1/1 с замкнутой переходной формулой."""
    times = np.array([0.0, 0.25, 1.0, 10.0])
    lam, mu = 3.0, 2.0
    result = _solve_two_state(lam, mu, times)
    occupied = lam / (lam + mu) * (1 - np.exp(-(lam + mu) * times))
    np.testing.assert_allclose(result[1], occupied, atol=1e-12)
    np.testing.assert_allclose(result[0], 1 - occupied, atol=1e-12)


def test_regression_published_mm11_example() -> None:
    """Сверяет M/M/1/1 с опубликованной вероятностью потерь Эрланга."""
    # Для стандартной M/M/1/1 при lambda=3 и mu=2 потери Эрланга равны 3/5.
    result = _solve_two_state(3.0, 2.0, np.array([0.0, 10.0]))
    assert math.isclose(result[1, -1], 0.6, abs_tol=1e-12)


def test_very_fast_service_keeps_system_almost_empty() -> None:
    """Проверяет почти пустую систему при быстром обслуживании."""
    result = _solve_two_state(1.0, 1_000_000.0, np.array([0.0, 0.01]))
    assert result[0, -1] > 0.999999
