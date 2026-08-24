"""Маршруты FastAPI-бекенда.

Содержит эндпоинты запуска моделирования СМО и справочника
допустимых значений перечислений.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from quick.domain import (
    CalculationEngine,
    CalculationMethod,
)
from quick.domain.enums import SystemType as DomainSystemType
from quick.services.systems import system_factory

from .converters import (
    STABILITY_METRIC_MAP,
    SYSTEM_MODE_MAP,
    SYSTEM_TYPE_MAP,
    build_computation_config,
    build_system_params,
    numpy_to_python,
)
from .schemas import (
    MetaResponse,
    SimulateRequest,
    SimulateResponse,
    SystemModeName,
)

router = APIRouter(prefix="/api/v1")

EXCLUDED_METRICS = {"probability"}


@router.get("/meta")
def get_meta() -> MetaResponse:
    """Возвращает справочник допустимых значений перечислений API.

    Returns:
        MetaResponse: Описания типов систем, режимов, методов и движков.
    """
    return MetaResponse(
        system_types={
            name.value: SYSTEM_TYPE_MAP[name].value for name in SYSTEM_TYPE_MAP
        },
        system_modes={
            name.value: SYSTEM_MODE_MAP[name].value for name in SYSTEM_MODE_MAP
        },
        calculation_methods={
            "analytical": CalculationMethod.ANALYTICAL.value,
            "numerical": CalculationMethod.NUMERICAL.value,
            "imitation": CalculationMethod.IMITATION.value,
        },
        calculation_engines={
            "numpy": CalculationEngine.NUMPY.value,
            "mpmath": CalculationEngine.MPMATH.value,
            "merged": CalculationEngine.MERGED.value,
        },
        stability_metrics={
            name.value: STABILITY_METRIC_MAP[name].value
            for name in STABILITY_METRIC_MAP
        },
    )


@router.post("/simulate")
def simulate(request: SimulateRequest) -> SimulateResponse:
    """Запускает моделирование СМО и возвращает результаты.

    Для переходного режима возвращает вероятности состояний во времени
    и метрики производительности, для стационарного — вектор вероятностей
    стационарных состояний.

    Args:
        request (SimulateRequest): Параметры системы и конфигурация вычислений.

    Returns:
        SimulateResponse: Результаты моделирования.

    Raises:
        HTTPException: 422 при некорректных параметрах,
            500 при внутренней ошибке вычислений.
    """
    config = build_computation_config(request.config)

    try:
        params = build_system_params(request)
        config.validate()
        params.validate()
    except ValueError as e:
        logger.warning(f"Некорректные параметры запроса: {e}")
        raise HTTPException(status_code=422, detail=str(e)) from e

    per_sensor = (
        request.per_sensor
        and params.calculation_params.system_type == DomainSystemType.MAP
    )

    try:
        system = system_factory(
            system_type=params.calculation_params.system_type,
            system_mode=params.calculation_params.system_mode,
            per_sensor=per_sensor,
            params=params,
            config=config,
        )
        system.calculate_probabilities()
    except ValueError as e:
        logger.warning(f"Ошибка при вычислении: {e}")
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("Внутренняя ошибка при моделировании")
        raise HTTPException(status_code=500, detail=str(e)) from e

    probabilities = numpy_to_python(system.probabilities)

    if request.system_mode == SystemModeName.TRANSIENT:
        results = system.calculate()
        metrics = {
            key: numpy_to_python(value)
            for key, value in results.items()
            if key not in EXCLUDED_METRICS
        }
        time_array = (
            numpy_to_python(params.transient_params.time_array)
            if params.transient_params is not None
            else None
        )
        return SimulateResponse(
            system_type=request.system_type,
            system_mode=request.system_mode,
            probabilities=probabilities,
            time_array=time_array,
            metrics=metrics,
        )

    return SimulateResponse(
        system_type=request.system_type,
        system_mode=request.system_mode,
        probabilities=probabilities,
    )
