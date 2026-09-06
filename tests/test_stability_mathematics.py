"""Тесты формул времени установления и устойчивости."""

import numpy as np
import pytest

from quick.domain.enums import StabilityMetric, StabilityVerdict
from quick.domain.params import StabilityParams
from quick.services.stability import (
    calculate_settling_time,
    calculate_stability_series,
    classify_stability,
    evaluate_stability,
)


def test_settling_time_uses_last_corridor_excursion() -> None:
    """Определяет время установления по последнему выходу из коридора."""
    times = np.arange(6, dtype=float)
    values = np.array([0.0, 1.0, 0.7, 0.99, 1.01, 1.0])
    assert calculate_settling_time(values, times, tolerance=0.05) == 3.0
    assert calculate_settling_time(np.ones(6), times, tolerance=0.05) == 0.0


def test_settling_time_filters_nan_and_uses_absolute_zero_corridor() -> None:
    """Игнорирует NaN и обрабатывает нулевое установившееся значение."""
    times = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([np.nan, 0.2, 0.01, 0.0])
    assert calculate_settling_time(values, times, tolerance=0.05) == 1.0


@pytest.mark.parametrize(
    ("values", "times", "message"),
    [
        (np.ones(2), np.ones(3), "совпадать"),
        (np.array([]), np.array([]), "пуст"),
        (np.array([np.nan]), np.array([0.0]), "не определена"),
    ],
)
def test_settling_time_rejects_undefined_inputs(
    values: np.ndarray, times: np.ndarray, message: str
) -> None:
    """Отклоняет несогласованные, пустые и неопределённые ряды."""
    with pytest.raises(ValueError, match=message):
        calculate_settling_time(values, times, 0.05)


def test_stability_integrals_margin_and_skipped_points() -> None:
    """Считает интегральные метрики без неопределённых точек."""
    params = StabilityParams(
        metric=StabilityMetric.THROUGHPUT,
        critical_level=1.0,
        settling_tolerance=0.01,
        stable_threshold=1.2,
    )
    result = evaluate_stability(
        np.array([np.nan, 2.0, 1.5, 1.5]),
        np.array([0.0, 1.0, 2.0, 3.0]),
        params,
    )
    assert result.skipped_points == 1
    assert result.coefficient == pytest.approx(1.75)
    assert result.margin == pytest.approx(75.0)
    assert result.area_above_critical == pytest.approx(0.75)
    assert result.holds_critical_level
    assert result.verdict is StabilityVerdict.STABLE


def test_stability_series_and_verdict_boundaries() -> None:
    """Проверяет границы классов устойчивости."""
    params = StabilityParams(critical_level=0.8, stable_threshold=1.2)
    np.testing.assert_allclose(
        calculate_stability_series(np.array([0.8, 1.0]), 0.8), [1.0, 1.25]
    )
    assert classify_stability(2.0, 0.7, params) is StabilityVerdict.UNSTABLE
    assert classify_stability(0.99, 0.9, params) is StabilityVerdict.UNSTABLE
    assert classify_stability(1.1, 0.9, params) is StabilityVerdict.BORDERLINE
    assert classify_stability(1.2, 0.9, params) is StabilityVerdict.STABLE
    assert classify_stability(np.nan, 0.9, params) is StabilityVerdict.UNSTABLE
    with pytest.raises(ValueError, match="положителен"):
        calculate_stability_series(np.ones(2), 0.0)


def test_stability_rejects_less_than_two_finite_points() -> None:
    """Отклоняет оценку без конечного временного интервала."""
    with pytest.raises(ValueError, match="менее чем в двух"):
        evaluate_stability(
            np.array([np.nan, 1.0]),
            np.array([0.0, 1.0]),
            StabilityParams(),
        )
