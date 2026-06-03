"""Генеративные тесты инвариантов вероятностной модели."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from quick.domain import ComputationConfig
from quick.domain.params import MultiSystemParams, TransientSystemParams
from quick.services.matrix_builders.multi import MultiServerMatrixBuilder
from quick.services.solvers.analytical.numpy_solver import (
    AnalyticalNumpyProbabilitySolver,
)


@settings(max_examples=30, deadline=None)
@given(
    lam=st.floats(min_value=0.05, max_value=10, allow_nan=False),
    mu=st.floats(min_value=0.05, max_value=10, allow_nan=False),
    nu=st.floats(min_value=0, max_value=10, allow_nan=False),
    capacity=st.integers(min_value=1, max_value=5),
    servers=st.integers(min_value=1, max_value=3),
)
def test_probability_invariants(
    lam: float, mu: float, nu: float, capacity: int, servers: int
) -> None:
    """Проверяет неотрицательность, нормировку и начальные условия."""
    base = MultiSystemParams(mu, nu, lam, capacity, servers)
    matrix = MultiServerMatrixBuilder(base).build()
    initial = np.zeros(capacity + servers + 1)
    initial[0] = 1.0
    transient = TransientSystemParams(
        np.linspace(0, 1, 5), np.zeros_like(initial), initial
    )
    probabilities = AnalyticalNumpyProbabilitySolver(
        transient, matrix, ComputationConfig(disable_cache=True)
    ).calculate()
    assert probabilities.min() >= -1e-10
    np.testing.assert_allclose(probabilities.sum(axis=0), 1.0, atol=1e-10)
    np.testing.assert_allclose(probabilities[:, 0], initial, atol=2e-10)
