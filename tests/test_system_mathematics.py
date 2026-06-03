"""Тесты формул и режимов многолинейной СМО."""

import numpy as np

from quick.domain import ComputationConfig
from quick.domain.enums import (
    CalculationMethod,
    StabilityMetric,
    SystemMode,
    SystemType,
)
from quick.domain.params import (
    CalculationParams,
    MultiSystemParams,
    StabilityParams,
    SystemParams,
    TransientSystemParams,
)
from quick.services.systems.factory import system_factory
from quick.services.systems.steady_state.multi import MultiServerSteadyStateSystem
from quick.services.systems.transient_state.multi import TransientMultiServerStateSystem


def _multi_system(mode: SystemMode) -> SystemParams:
    """Создаёт параметры многолинейной СМО в заданном режиме."""
    base = MultiSystemParams(2.0, 0.5, 1.0, max_customers=2, processor_count=1)
    transient = None
    if mode is SystemMode.TRANSIENT:
        initial = np.array([1.0, 0.0, 0.0, 0.0])
        transient = TransientSystemParams(
            np.linspace(0, 2, 9), np.zeros_like(initial), initial
        )
    return SystemParams(
        base,
        CalculationParams(SystemType.MULTI, mode),
        transient,
        StabilityParams(critical_level=0.5),
    )


def test_multi_steady_state_is_normalized_and_metrics_obey_identities() -> None:
    """Проверяет нормировку и тождества пропускной способности."""
    system = system_factory(
        SystemType.MULTI, SystemMode.STEADY, params=_multi_system(SystemMode.STEADY)
    )
    assert isinstance(system, MultiServerSteadyStateSystem)
    probabilities = system.probabilities
    np.testing.assert_allclose(probabilities.sum(), 1.0)
    assert probabilities.min() >= 0
    np.testing.assert_allclose(
        system.calculate_absolute_throughput(),
        system.lambda_rate * system.calculate_relative_throughput(),
    )
    np.testing.assert_allclose(
        system.calculate_rejection_probability(),
        1.0 - system.calculate_relative_throughput(),
    )
    assert set(system.calculate()) == {
        "probability",
        "throughput",
        "absolute_throughput",
        "relative_throughput",
        "avg_system_length",
        "rejection_probability",
        "stability_coefficient",
    }


def test_multi_transient_system_runs_all_characteristics() -> None:
    """Рассчитывает все характеристики переходной СМО."""
    params = _multi_system(SystemMode.TRANSIENT)
    config = ComputationConfig(
        calculation_method=CalculationMethod.NUMERICAL, disable_cache=True
    )
    system = system_factory(
        SystemType.MULTI,
        SystemMode.TRANSIENT,
        params=params,
        config=config,
    )
    assert isinstance(system, TransientMultiServerStateSystem)
    results = system.calculate()
    np.testing.assert_allclose(results["probability"].sum(axis=0), 1.0, atol=1e-6)
    assert system.log_average_lambda() == 1.0
    assert system.beta == 0.25
    assert system.rho == 0.5
    assert system.evaluate_stability().coefficient >= 0


def test_throughput_can_be_selected_as_stability_metric() -> None:
    """Использует пропускную способность как метрику устойчивости."""
    params = _multi_system(SystemMode.TRANSIENT)
    params.stability_params = StabilityParams(
        metric=StabilityMetric.THROUGHPUT,
        critical_level=0.1,
    )
    system = system_factory(
        SystemType.MULTI,
        SystemMode.TRANSIENT,
        params=params,
        config=ComputationConfig(calculation_method=CalculationMethod.NUMERICAL),
    )
    assert isinstance(system, TransientMultiServerStateSystem)
    np.testing.assert_allclose(
        system.calculate_stability_base_metric(), system.calculate_throughput()
    )
