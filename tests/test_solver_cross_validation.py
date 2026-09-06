"""Перекрёстные тесты аналитических, численных и имитационных решений."""

import numpy as np
import pytest

from quick.domain import (
    ComputationConfig,
    MergedComputationConfig,
    MpmathComputationConfig,
)
from quick.domain.enums import CalculationEngine, SystemMode, SystemType
from quick.domain.params import (
    CalculationParams,
    ImitationSystemParams,
    MultiSystemParams,
    SimulationParams,
    TimeSeriesBaseSystemParams,
    TransientSystemParams,
)
from quick.services.matrix_builders.multi import MultiServerMatrixBuilder
from quick.services.solvers.analytical.merged_solver import (
    AnalyticalMergedProbabilitySolver,
)
from quick.services.solvers.analytical.mpmath_solver import (
    AnalyticalMpmathProbabilitySolver,
)
from quick.services.solvers.analytical.numpy_solver import (
    AnalyticalNumpyProbabilitySolver,
)
from quick.services.solvers.imitation.multi import (
    MultiServerImitationProbabilitySolver,
)
from quick.services.solvers.numerical.solver import NumericalProbabilitySolver

PARAMETER_GRID = [(0.5, 1.0, 0.0), (1.5, 2.0, 0.4), (3.0, 1.5, 1.0)]


def _problem(
    rates: tuple[float, float, float], trajectories: int = 1
) -> tuple[TransientSystemParams, np.ndarray, MultiSystemParams, ImitationSystemParams]:
    """Создаёт общие входные данные для всех семейств решателей."""
    lam, mu, nu = rates
    base = MultiSystemParams(mu, nu, lam, max_customers=2, processor_count=1)
    times = np.linspace(0.0, 2.0, 9)
    initial = np.array([1.0, 0.0, 0.0, 0.0])
    transient = TransientSystemParams(times, np.zeros(4), initial)
    calculation = CalculationParams(SystemType.MULTI, SystemMode.TRANSIENT)
    system = ImitationSystemParams(
        base_params=base,
        transient_params=transient,
        calculation_params=calculation,
        simulation_params=SimulationParams(trajectories=trajectories, seed=20260912),
        time_series_params=TimeSeriesBaseSystemParams(),
    )
    return transient, MultiServerMatrixBuilder(base).build(), base, system


@pytest.mark.parametrize("rates", PARAMETER_GRID)
def test_analytical_and_numerical_agree_on_parameter_grid(
    rates: tuple[float, float, float],
) -> None:
    """Сверяет аналитическое решение и RK45 на общей сетке."""
    transient, matrix, _, _ = _problem(rates)
    analytical = AnalyticalNumpyProbabilitySolver(
        transient, matrix, ComputationConfig(disable_cache=True)
    ).calculate()
    numerical = NumericalProbabilitySolver(transient, matrix).calculate()
    np.testing.assert_allclose(analytical, numerical, atol=2e-4, rtol=2e-4)


def test_numpy_and_mpmath_agree_with_declared_tolerance() -> None:
    """Сверяет аналитические решения NumPy и mpmath."""
    transient, matrix, _, _ = _problem((1.5, 2.0, 0.4))
    numpy_result = AnalyticalNumpyProbabilitySolver(
        transient, matrix, ComputationConfig(disable_cache=True)
    ).calculate()
    mpmath_result = AnalyticalMpmathProbabilitySolver(
        transient,
        matrix,
        MpmathComputationConfig(precision=60, disable_cache=True),
    ).calculate()
    np.testing.assert_allclose(numpy_result, mpmath_result, atol=1e-11, rtol=1e-11)


def test_merged_solver_matches_high_precision_solution() -> None:
    """Сверяет гибридный решатель с эталонным решением mpmath."""
    transient, matrix, _, _ = _problem((1.5, 2.0, 0.4))
    config = MergedComputationConfig(
        calculation_engine=CalculationEngine.MERGED,
        precision=60,
        tolerance=1e-10,
        disable_cache=True,
    )
    merged = AnalyticalMergedProbabilitySolver(transient, matrix, config)
    result = merged.calculate()
    reference = AnalyticalMpmathProbabilitySolver(
        transient,
        matrix,
        MpmathComputationConfig(precision=60, disable_cache=True),
    ).calculate()
    np.testing.assert_allclose(result, reference, atol=1e-10, rtol=1e-10)


def test_merged_solver_detects_value_and_normalization_errors() -> None:
    """Находит некорректные значения и нарушения нормировки."""
    transient, matrix, _, _ = _problem((1.5, 2.0, 0.4))
    solver = AnalyticalMergedProbabilitySolver(
        transient,
        matrix,
        MergedComputationConfig(tolerance=1e-6, disable_cache=True),
    )
    valid = np.repeat(np.eye(4)[:, :, np.newaxis], 3, axis=2)
    assert np.all(solver._get_invalid_indices(valid) == -1)
    invalid = valid.copy()
    invalid[0, 0, 1] = 1.1
    invalid[1, 2, 2] = -0.1
    indices = solver._get_invalid_indices(invalid, check_sum=False)
    assert indices[0, 0] == 1
    assert indices[1, 2] == 2
    normalized_indices = solver._get_invalid_indices(invalid, check_sum=True)
    assert np.all(normalized_indices[:, 0] >= 1)


@pytest.mark.monte_carlo
@pytest.mark.parametrize("rates", PARAMETER_GRID)
def test_monte_carlo_tracks_analytical_solution(
    rates: tuple[float, float, float],
) -> None:
    """Проверяет оценки Monte-Carlo в статистическом коридоре."""
    trajectories = 5_000
    transient, matrix, _, imitation = _problem(rates, trajectories=trajectories)
    exact = AnalyticalNumpyProbabilitySolver(
        transient, matrix, ComputationConfig(disable_cache=True)
    ).calculate()
    sampled = MultiServerImitationProbabilitySolver(imitation).calculate()
    # Шестисигмовый коридор биномиальной выборки с минимумом для редких состояний.
    sigma = np.sqrt(np.maximum(exact * (1.0 - exact), 1e-3) / trajectories)
    assert np.all(np.abs(sampled - exact) <= 6.0 * sigma + 0.012)
