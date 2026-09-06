"""Тесты генераторов, решателей и характеристик MAP-СМО."""

import numpy as np
import pytest

from quick.domain import ComputationConfig
from quick.domain.enums import CalculationMethod, SystemMode, SystemType
from quick.domain.params import (
    CalculationParams,
    ImitationSystemParams,
    MAPSystemParams,
    SimulationParams,
    SystemParams,
    TimeSeriesBaseSystemParams,
    TransientSystemParams,
)
from quick.services.matrix_builders.map import MultiSensorMAPServerMatrixBuilder
from quick.services.solvers.imitation.factory import imitation_solvers_factory
from quick.services.solvers.imitation.map import MAPImitationProbabilitySolver
from quick.services.systems.factory import system_factory
from quick.services.systems.steady_state.map import MAPSteadyStateSystem
from quick.services.systems.transient_state.map import TransientMAPServerStateSystem


def _map_params(mode: SystemMode = SystemMode.STEADY) -> SystemParams:
    """Создаёт корректные параметры двухфазной MAP-СМО."""
    base = MAPSystemParams(
        mu_rate=4.0,
        nu_rate=1.0,
        lambda_rate=np.array([2.0, 3.0]),
        max_customers=3,
        p_rate=np.array([[0.0, 0.2], [0.1, 0.0]]),
        q_rate=np.array([[0.8, 0.0], [0.0, 0.9]]),
        sensor_count=2,
    )
    transient = None
    if mode is SystemMode.TRANSIENT:
        initial = np.array([0.6, 0.4, 0.0, 0.0, 0.0, 0.0])
        transient = TransientSystemParams(
            np.linspace(0.0, 1.0, 6), np.zeros_like(initial), initial
        )
    return SystemParams(base, CalculationParams(SystemType.MAP, mode), transient)


def _map_base(params: SystemParams) -> MAPSystemParams:
    """Возвращает базовые параметры MAP-СМО с уточнённым типом."""
    assert isinstance(params.base_params, MAPSystemParams)
    return params.base_params


def test_map_blocks_and_generator_conservation() -> None:
    """Строит D0, D1 и сохраняющий вероятность MAP-генератор."""
    params = _map_params()
    builder = MultiSensorMAPServerMatrixBuilder(_map_base(params))
    np.testing.assert_allclose(
        builder._d_0_matrix_generator(), [[-2.0, 0.4], [0.3, -3.0]]
    )
    np.testing.assert_allclose(
        builder._d_1_matrix_generator(), [[1.6, 0.0], [0.0, 2.7]]
    )
    generator = builder.build()
    assert generator.shape == (6, 6)
    np.testing.assert_allclose(generator.sum(axis=0), 0.0, atol=1e-15)


def test_map_stationary_distribution_solves_balance_equations() -> None:
    """Решает уравнения баланса MAP с нормированным распределением."""
    params = _map_params()
    system = system_factory(
        SystemType.MAP,
        SystemMode.STEADY,
        params=params,
        config=ComputationConfig(disable_cache=True),
    )
    assert isinstance(system, MAPSteadyStateSystem)
    probabilities = system.probabilities[:, 0]
    generator = MultiSensorMAPServerMatrixBuilder(_map_base(params)).build()
    assert probabilities.min() >= 0
    np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(generator @ probabilities, 0.0, atol=1e-11)


def test_map_transient_characteristics_obey_model_identities() -> None:
    """Проверяет тождества между переходными характеристиками MAP."""
    params = _map_params(SystemMode.TRANSIENT)
    system = system_factory(
        SystemType.MAP,
        SystemMode.TRANSIENT,
        params=params,
        config=ComputationConfig(
            calculation_method=CalculationMethod.NUMERICAL,
            disable_cache=True,
        ),
    )
    assert isinstance(system, TransientMAPServerStateSystem)
    results = system.calculate()
    np.testing.assert_allclose(results["probability"].sum(axis=0), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        results["loss_probability"],
        results["rejection_probability"] + results["quit_probability"],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        results["service_probability"], 1.0 - results["loss_probability"]
    )
    np.testing.assert_allclose(
        results["throughput"],
        results["service_probability"] * results["input_intensity"],
    )
    np.testing.assert_allclose(results["phase_distribution"].sum(axis=1), 1.0)
    np.testing.assert_allclose(
        results["loss_flow_intensity"], results["avg_system_length"]
    )
    assert results["quit_probability"][0] == 0.0
    assert system.log_average_lambda() > 0


def test_per_sensor_map_components_add_up_to_aggregate() -> None:
    """Получает общие значения суммированием вкладов датчиков."""
    params = _map_params(SystemMode.TRANSIENT)
    config = ComputationConfig(calculation_method=CalculationMethod.NUMERICAL)
    aggregate = system_factory(
        SystemType.MAP, SystemMode.TRANSIENT, params=params, config=config
    )
    components = system_factory(
        SystemType.MAP,
        SystemMode.TRANSIENT,
        per_sensor=True,
        params=params,
        config=config,
    )
    assert isinstance(aggregate, TransientMAPServerStateSystem)
    assert isinstance(components, TransientMAPServerStateSystem)
    aggregate.calculate_probabilities()
    components._probabilities = aggregate.probabilities
    np.testing.assert_allclose(
        components.calculate_avg_system_length().sum(axis=1),
        aggregate.calculate_avg_system_length(),
    )
    np.testing.assert_allclose(
        components.calculate_loss_flow_intensity().sum(axis=1),
        aggregate.calculate_loss_flow_intensity(),
    )


def test_map_monte_carlo_uses_same_generator_and_normalizes() -> None:
    """Использует общий генератор в нормированной MAP-имитации."""
    params = _map_params(SystemMode.TRANSIENT)
    imitation_params = ImitationSystemParams(
        base_params=params.base_params,
        calculation_params=params.calculation_params,
        transient_params=params.transient_params,
        simulation_params=SimulationParams(trajectories=500, seed=42),
        time_series_params=TimeSeriesBaseSystemParams(),
    )
    solver = imitation_solvers_factory(imitation_params)
    assert isinstance(solver, MAPImitationProbabilitySolver)
    q_row = solver._get_q_row(0.0)
    expected = MultiSensorMAPServerMatrixBuilder(_map_base(params)).build().T
    np.testing.assert_allclose(q_row, expected)
    transitions = solver._get_transitions(0, 0.0)
    assert transitions
    assert all(rate > 0 and state != 0 for rate, state in transitions)
    probabilities = solver.calculate()
    np.testing.assert_allclose(probabilities.sum(axis=0), 1.0)


def test_stationary_linear_solver_validates_shape_and_handles_singular_matrix() -> None:
    """Проверяет форму матрицы и обработку вырожденной системы."""
    system = MAPSteadyStateSystem(
        _map_params(),
        system_behavior=system_factory(
            SystemType.MAP,
            SystemMode.STEADY,
            params=_map_params(),
        ).system_behavior.__class__,
    )
    with pytest.raises(ValueError, match="двумерной"):
        system._solve_stationary_system(np.zeros(3))
    with pytest.raises(ValueError, match="квадратной"):
        system._solve_stationary_system(np.zeros((2, 3)))
    result = system._solve_stationary_system(np.zeros((2, 2)))
    assert result.min() >= 0
    np.testing.assert_allclose(result.sum(), 1.0)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"nu_rate": 5.0}, "ν <= μ"),
        ({"sensor_count": 3}, "числом MAP-фаз"),
        ({"p_rate": np.eye(2)}, "диагональ"),
        ({"q_rate": np.zeros((2, 2))}, "не может быть нулевой"),
    ],
)
def test_map_parameter_rejections(change: dict[str, object], message: str) -> None:
    """Отклоняет параметры вне области применимости MAP-модели."""
    base = _map_params().base_params
    for name, value in change.items():
        setattr(base, name, value)
    with pytest.raises(ValueError, match=message):
        base.validate()
