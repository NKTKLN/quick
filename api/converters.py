"""Преобразование Pydantic-схем API в доменные объекты пакета quick.

Содержит функции конвертации запроса SimulateRequest в конфигурацию
вычислений и параметры системы, пригодные для передачи в system_factory.
"""

import json
from typing import Any

import numpy as np

from quick.domain import (
    CalculationEngine,
    CalculationMethod,
    ComputationConfig,
    MergedComputationConfig,
    MpmathComputationConfig,
    StabilityMetric,
    SystemMode,
    SystemType,
)
from quick.domain.params import (
    BaseSystemParams,
    CalculationParams,
    ImitationSystemParams,
    MAPSystemParams,
    MultiSystemParams,
    SimulationParams,
    StabilityParams,
    SystemParams,
    TimeSeries,
    TimeSeriesBaseSystemParams,
    TransientSystemParams,
)

from .schemas import (
    CalculationEngineName,
    CalculationMethodName,
    ComputationConfigSchema,
    SimulateRequest,
    StabilityMetricName,
    SystemModeName,
    SystemTypeName,
)

SYSTEM_TYPE_MAP = {
    SystemTypeName.MULTI: SystemType.MULTI,
    SystemTypeName.MAP: SystemType.MAP,
}

SYSTEM_MODE_MAP = {
    SystemModeName.TRANSIENT: SystemMode.TRANSIENT,
    SystemModeName.STEADY: SystemMode.STEADY,
}

CALCULATION_METHOD_MAP = {
    CalculationMethodName.ANALYTICAL: CalculationMethod.ANALYTICAL,
    CalculationMethodName.NUMERICAL: CalculationMethod.NUMERICAL,
    CalculationMethodName.IMITATION: CalculationMethod.IMITATION,
}

STABILITY_METRIC_MAP = {
    StabilityMetricName.SERVICE_PROBABILITY: StabilityMetric.SERVICE_PROBABILITY,
    StabilityMetricName.THROUGHPUT: StabilityMetric.THROUGHPUT,
}

CALCULATION_ENGINE_MAP = {
    CalculationEngineName.NUMPY: CalculationEngine.NUMPY,
    CalculationEngineName.MPMATH: CalculationEngine.MPMATH,
    CalculationEngineName.MERGED: CalculationEngine.MERGED,
}


def build_computation_config(schema: ComputationConfigSchema) -> ComputationConfig:
    """Создаёт конфигурацию вычислений из схемы запроса.

    Args:
        schema (ComputationConfigSchema): Схема конфигурации из запроса.

    Returns:
        ComputationConfig: Доменная конфигурация вычислений.
    """
    method = CALCULATION_METHOD_MAP[schema.calculation_method]
    engine = CALCULATION_ENGINE_MAP[schema.calculation_engine]

    if method != CalculationMethod.ANALYTICAL:
        return ComputationConfig(
            calculation_engine=CalculationEngine.NUMPY,
            calculation_method=method,
            disable_cache=schema.disable_cache,
            compute_only_last_state=schema.compute_only_last_state,
        )

    if engine == CalculationEngine.MPMATH:
        return MpmathComputationConfig(
            calculation_engine=engine,
            calculation_method=method,
            disable_cache=schema.disable_cache,
            compute_only_last_state=schema.compute_only_last_state,
            precision=schema.precision,
        )
    if engine == CalculationEngine.MERGED:
        return MergedComputationConfig(
            calculation_engine=engine,
            calculation_method=method,
            disable_cache=schema.disable_cache,
            compute_only_last_state=schema.compute_only_last_state,
            precision=schema.precision,
            tolerance=schema.tolerance,
        )
    return ComputationConfig(
        calculation_engine=engine,
        calculation_method=method,
        disable_cache=schema.disable_cache,
        compute_only_last_state=schema.compute_only_last_state,
    )


def _build_base_params(request: SimulateRequest) -> BaseSystemParams:
    """Создаёт базовые параметры системы из запроса.

    Args:
        request (SimulateRequest): Запрос на моделирование.

    Returns:
        BaseSystemParams: Доменные параметры системы.

    Raises:
        ValueError: Если параметры не соответствуют типу системы.
    """
    if request.system_type == SystemTypeName.MULTI:
        if request.processor_count is None:
            raise ValueError("processor_count обязателен для системы типа multi.")
        if not isinstance(request.lambda_rate, int | float):
            raise ValueError("lambda_rate должен быть скаляром для системы типа multi.")
        return MultiSystemParams(
            mu_rate=request.mu_rate,
            nu_rate=request.nu_rate,
            lambda_rate=float(request.lambda_rate),
            max_customers=request.max_customers,
            processor_count=request.processor_count,
        )

    if request.map_params is None:
        raise ValueError("map_params обязательны для системы типа map.")
    if not isinstance(request.lambda_rate, list):
        raise ValueError("lambda_rate должен быть списком для системы типа map.")
    return MAPSystemParams(
        mu_rate=request.mu_rate,
        nu_rate=request.nu_rate,
        lambda_rate=np.asarray(request.lambda_rate, dtype=np.float64),
        max_customers=request.max_customers,
        p_rate=np.asarray(request.map_params.p_rate, dtype=np.float64),
        q_rate=np.asarray(request.map_params.q_rate, dtype=np.float64),
        sensor_count=request.map_params.sensor_count,
    )


def _build_transient_params(
    request: SimulateRequest,
) -> TransientSystemParams | None:
    """Создаёт параметры переходного режима из запроса.

    Args:
        request (SimulateRequest): Запрос на моделирование.

    Returns:
        TransientSystemParams | None: Параметры переходного режима или None
        для стационарного режима.

    Raises:
        ValueError: Если параметры переходного режима отсутствуют.
    """
    if request.system_mode != SystemModeName.TRANSIENT:
        return None
    if request.transient_params is None:
        raise ValueError("transient_params обязательны в переходном режиме.")

    initial_probabilities = np.asarray(
        request.transient_params.initial_probabilities, dtype=np.float64
    )
    return TransientSystemParams(
        time_array=np.asarray(request.transient_params.time_array, dtype=np.float64),
        state_variables=np.zeros_like(initial_probabilities),
        initial_probabilities=initial_probabilities,
    )


def _build_time_series_params(
    request: SimulateRequest,
) -> TimeSeriesBaseSystemParams:
    """Создаёт зависящие от времени параметры СМО из запроса.

    Args:
        request (SimulateRequest): Запрос на моделирование.

    Returns:
        TimeSeriesBaseSystemParams: Параметры, зависящие от времени.
    """
    if request.time_series_params is None:
        return TimeSeriesBaseSystemParams()

    def to_series(schema: Any) -> TimeSeries | None:
        if schema is None:
            return None
        return TimeSeries(
            times=np.asarray(schema.times, dtype=np.float64),
            values=np.asarray(schema.values, dtype=np.float64),
        )

    return TimeSeriesBaseSystemParams(
        lambda_rate=to_series(request.time_series_params.lambda_rate),
        mu_rate=to_series(request.time_series_params.mu_rate),
        nu_rate=to_series(request.time_series_params.nu_rate),
    )


def _build_stability_params(request: SimulateRequest) -> StabilityParams:
    """Создаёт параметры оценки устойчивости из запроса.

    Args:
        request (SimulateRequest): Запрос на моделирование.

    Returns:
        StabilityParams: Параметры оценки устойчивости; при отсутствии
            блока в запросе — значения по умолчанию.
    """
    schema = request.stability

    if schema is None:
        return StabilityParams()

    return StabilityParams(
        metric=STABILITY_METRIC_MAP[schema.metric],
        critical_level=schema.critical_level,
        settling_tolerance=schema.settling_tolerance,
        stable_threshold=schema.stable_threshold,
    )


def build_system_params(request: SimulateRequest) -> SystemParams:
    """Создаёт полные параметры системы из запроса.

    Args:
        request (SimulateRequest): Запрос на моделирование.

    Returns:
        SystemParams: Доменные параметры системы
        (ImitationSystemParams для имитационного метода).
    """
    base_params = _build_base_params(request)
    transient_params = _build_transient_params(request)
    calculation_params = CalculationParams(
        system_type=SYSTEM_TYPE_MAP[request.system_type],
        system_mode=SYSTEM_MODE_MAP[request.system_mode],
        sensor_index=request.sensor_index,
    )
    stability_params = _build_stability_params(request)

    if request.config.calculation_method == CalculationMethodName.IMITATION:
        simulation_schema = request.simulation_params
        simulation_params = SimulationParams(
            trajectories=simulation_schema.trajectories
            if simulation_schema
            else 10_000,
            seed=simulation_schema.seed if simulation_schema else None,
        )
        return ImitationSystemParams(
            base_params=base_params,
            transient_params=transient_params,
            calculation_params=calculation_params,
            stability_params=stability_params,
            simulation_params=simulation_params,
            time_series_params=_build_time_series_params(request),
        )

    return SystemParams(
        base_params=base_params,
        transient_params=transient_params,
        calculation_params=calculation_params,
        stability_params=stability_params,
    )


def numpy_to_python(obj: Any) -> Any:
    """Рекурсивно преобразует numpy-объекты в стандартные Python-типы.

    Преобразует:
    - numpy.ndarray → list
    - numpy числа → Python числа
    - numpy bool → bool
    - NaN и Inf → None

    Args:
        obj (Any): Объект, который может содержать numpy-типы.

    Returns:
        Any: Объект, содержащий только стандартные Python-типы.
    """
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, np.integer | np.floating | np.bool_):
        return _sanitize(obj.item())
    if isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    if isinstance(obj, list | tuple):
        return [numpy_to_python(item) for item in obj]
    return _sanitize(obj)


def _sanitize(obj: Any) -> Any:
    """Заменяет NaN и Inf на None для совместимости с JSON.

    Args:
        obj (Any): Объект для очистки.

    Returns:
        Any: Объект без NaN и Inf.
    """
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, list):
        return [_sanitize(item) for item in obj]
    return obj


def ensure_json_safe(obj: Any) -> Any:
    """Проверяет, что объект сериализуем в JSON без NaN и Inf.

    Args:
        obj (Any): Объект для проверки.

    Returns:
        Any: Исходный объект.

    Raises:
        ValueError: Если объект содержит несериализуемые значения.
    """
    json.dumps(obj, allow_nan=False)
    return obj
