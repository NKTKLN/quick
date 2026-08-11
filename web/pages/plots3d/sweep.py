"""Модуль развёртки характеристик СМО по значению параметра.

Содержит описание разворачиваемого параметра, UI для задания его сетки и
функцию прогона расчёта по всем точкам сетки. Результатом является набор
метрик формы [точка сетки, время], пригодный для построения поверхности.

Основные функции:
    render_parameter_controls() — UI выбора параметра и его сетки.

    run_parameter_sweep() — выполняет расчёт системы для каждой точки сетки
    и собирает значения метрик.
"""

from dataclasses import dataclass, replace

import numpy as np
import streamlit as st
from numpy.typing import NDArray

from quick.domain import ComputationConfig, SystemType
from quick.domain.params import SystemParams
from quick.services.systems import system_factory
from quick.utils.progress import Progress

PARAMETER_MU = "Интенсивность обслуживания (μ)"
PARAMETER_NU = "Интенсивность ухода нетерпеливых заявок (ν)"
PARAMETER_LAMBDA = "Интенсивность поступления заявок (λ)"
PARAMETER_LAMBDA_SENSOR = "Интенсивность поступления заявок λ отдельного датчика"
PARAMETER_LAMBDA_SCALE = "Масштаб всех интенсивностей λ"

MIN_GRID_POINTS = 3
MAX_GRID_POINTS = 40
DEFAULT_GRID_POINTS = 12


@dataclass(frozen=True)
class SweepParameter:
    """Параметр, по которому строится третья ось графика.

    Attributes:
        name (str): Отображаемое имя параметра.
        kind (str): Тип параметра: "mu", "nu", "lambda", "lambda_sensor"
            или "lambda_scale".
        grid (NDArray[np.float64]): Значения параметра, для которых
            выполняется расчёт.
        sensor_index (int | None): Номер датчика для типа "lambda_sensor".
    """

    name: str
    kind: str
    grid: NDArray[np.float64]
    sensor_index: int | None = None

    @property
    def axis_title(self) -> str:
        """Подпись третьей оси графика.

        Returns:
            str: Название параметра для оси Y.
        """
        if self.kind == "lambda_sensor":
            return f"λ датчика {self.sensor_index}"
        if self.kind == "lambda_scale":
            return "Масштаб λ"
        return {"mu": "μ", "nu": "ν"}.get(self.kind, "λ")

    @property
    def cache_key(self) -> tuple:
        """Ключ кэширования результатов развёртки.

        Returns:
            tuple: Кортеж, однозначно определяющий развёртку.
        """
        return (self.kind, self.sensor_index, tuple(self.grid.tolist()))


def _available_parameters(params: SystemParams) -> list[str]:
    """Возвращает список параметров, доступных для развёртки.

    Args:
        params (SystemParams): Параметры системы.

    Returns:
        list[str]: Названия параметров.
    """
    options = [PARAMETER_NU, PARAMETER_MU]

    if params.calculation_params.system_type == SystemType.MAP:
        options += [PARAMETER_LAMBDA_SENSOR, PARAMETER_LAMBDA_SCALE]
    else:
        options.append(PARAMETER_LAMBDA)

    return options


def _default_value(params: SystemParams, name: str, sensor_index: int) -> float:
    """Возвращает текущее значение параметра для подстановки в границы сетки.

    Args:
        params (SystemParams): Параметры системы.
        name (str): Название параметра.
        sensor_index (int): Номер датчика для λ отдельного датчика.

    Returns:
        float: Текущее значение параметра.
    """
    base_params = params.base_params

    if name == PARAMETER_MU:
        return float(base_params.mu_rate)
    if name == PARAMETER_NU:
        return float(base_params.nu_rate)
    if name == PARAMETER_LAMBDA_SCALE:
        return 1.0
    if name == PARAMETER_LAMBDA_SENSOR:
        return float(np.asarray(base_params.lambda_rate)[sensor_index])

    return float(np.asarray(base_params.lambda_rate).reshape(-1)[0])


def render_parameter_controls(params: SystemParams, key_prefix: str) -> SweepParameter:
    """Отображает UI выбора разворачиваемого параметра и его сетки.

    Args:
        params (SystemParams): Параметры системы.
        key_prefix (str): Префикс ключей Streamlit-виджетов, уникальный
            для каждого графика.

    Returns:
        SweepParameter: Описание параметра и сетки его значений.
    """
    parameter_name = st.selectbox(
        "Параметр развёртки",
        _available_parameters(params),
        key=f"{key_prefix}_parameter",
    )

    sensor_index = 0
    if parameter_name == PARAMETER_LAMBDA_SENSOR:
        sensor_count = int(np.asarray(params.base_params.lambda_rate).shape[0])
        sensor_index = int(
            st.number_input(
                "Номер датчика",
                min_value=0,
                max_value=sensor_count - 1,
                value=0,
                key=f"{key_prefix}_sensor",
            )
        )

    current_value = _default_value(params, parameter_name, sensor_index)

    col1, col2, col3 = st.columns(3)
    with col1:
        min_value = st.number_input(
            "Минимум",
            min_value=0.0,
            value=max(current_value * 0.5, 1e-9),
            format="%.6f",
            key=f"{key_prefix}_min",
        )
    with col2:
        max_value = st.number_input(
            "Максимум",
            min_value=0.0,
            value=max(current_value * 1.5, current_value + 1.0),
            format="%.6f",
            key=f"{key_prefix}_max",
        )
    with col3:
        point_count = int(
            st.number_input(
                "Число точек",
                min_value=MIN_GRID_POINTS,
                max_value=MAX_GRID_POINTS,
                value=DEFAULT_GRID_POINTS,
                key=f"{key_prefix}_points",
            )
        )

    kinds = {
        PARAMETER_MU: "mu",
        PARAMETER_NU: "nu",
        PARAMETER_LAMBDA: "lambda",
        PARAMETER_LAMBDA_SENSOR: "lambda_sensor",
        PARAMETER_LAMBDA_SCALE: "lambda_scale",
    }

    return SweepParameter(
        name=parameter_name,
        kind=kinds[parameter_name],
        grid=np.linspace(min_value, max_value, point_count, dtype=np.float64),
        sensor_index=sensor_index
        if parameter_name == PARAMETER_LAMBDA_SENSOR
        else None,
    )


def apply_parameter(
    params: SystemParams, parameter: SweepParameter, value: float
) -> SystemParams:
    """Создаёт копию параметров системы с изменённым значением параметра.

    Args:
        params (SystemParams): Исходные параметры системы.
        parameter (SweepParameter): Описание разворачиваемого параметра.
        value (float): Значение параметра для текущей точки сетки.

    Returns:
        SystemParams: Копия параметров с подставленным значением.

    Raises:
        ValueError: Если тип параметра не поддерживается.
    """
    base_params = params.base_params

    match parameter.kind:
        case "mu":
            new_base_params = replace(base_params, mu_rate=value)
        case "nu":
            new_base_params = replace(base_params, nu_rate=value)
        case "lambda":
            new_base_params = replace(base_params, lambda_rate=value)
        case "lambda_scale":
            lambda_rate = np.asarray(base_params.lambda_rate, dtype=np.float64) * value
            new_base_params = replace(base_params, lambda_rate=lambda_rate)
        case "lambda_sensor":
            lambda_rate = np.array(base_params.lambda_rate, dtype=np.float64)
            lambda_rate[parameter.sensor_index] = value
            new_base_params = replace(base_params, lambda_rate=lambda_rate)
        case _:
            raise ValueError(f"Неизвестный параметр развёртки: {parameter.kind}")

    return replace(params, base_params=new_base_params)


def _metric_row(
    metrics: dict[str, NDArray[np.float64]], metric_key: str, time_count: int
) -> NDArray[np.float64] | None:
    """Извлекает значения метрики, пригодные для строки поверхности.

    Args:
        metrics (dict[str, NDArray[np.float64]]): Результаты расчёта.
        metric_key (str): Ключ метрики.
        time_count (int): Ожидаемое число временных точек.

    Returns:
        NDArray[np.float64] | None: Значения метрики формы [time_count]
            или None, если метрика не является одномерной по времени.
    """
    values = np.asarray(metrics.get(metric_key), dtype=np.float64)

    if values.ndim != 1 or values.shape[0] != time_count:
        return None

    return values


def run_parameter_sweep(
    config: ComputationConfig,
    params: SystemParams,
    parameter: SweepParameter,
    metric_keys: list[str],
    time_count: int,
) -> dict[str, NDArray[np.float64]]:
    """Выполняет расчёт системы для каждой точки сетки параметра.

    Точки, для которых параметры невалидны или расчёт завершился ошибкой,
    заполняются значениями NaN и не прерывают развёртку.

    Args:
        config (ComputationConfig): Конфигурация вычислений.
        params (SystemParams): Базовые параметры системы.
        parameter (SweepParameter): Описание параметра и сетки значений.
        metric_keys (list[str]): Ключи метрик, которые нужно собрать.
        time_count (int): Число временных точек расчёта.

    Returns:
        dict[str, NDArray[np.float64]]: Значения метрик формы
            [число точек сетки, time_count].
    """
    surfaces: dict[str, NDArray[np.float64]] = {
        metric_key: np.full((parameter.grid.shape[0], time_count), np.nan)
        for metric_key in metric_keys
    }
    failed_points: list[float] = []

    for index, value in enumerate(
        Progress.wrap(list(parameter.grid), f"Развёртка по {parameter.axis_title}")
    ):
        try:
            point_params = apply_parameter(params, parameter, float(value))
            point_params.validate()

            system = system_factory(
                system_mode=point_params.calculation_params.system_mode,
                system_type=point_params.calculation_params.system_type,
                params=point_params,
                config=config,
            )
            metrics = system.calculate()
        except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError):
            failed_points.append(float(value))
            continue

        for metric_key in metric_keys:
            row = _metric_row(metrics, metric_key, time_count)
            if row is not None:
                surfaces[metric_key][index] = row

    if failed_points:
        st.warning(
            f"⚠️ Расчёт не удался для {len(failed_points)} точек сетки "
            f"(например, {parameter.axis_title} = {failed_points[0]:.6g}); "
            "на поверхности они показаны как разрывы."
        )

    return surfaces


def get_sweep(
    config: ComputationConfig,
    params: SystemParams,
    parameter: SweepParameter,
    metric_keys: list[str],
    time_count: int,
    cache: dict,
) -> dict[str, NDArray[np.float64]]:
    """Возвращает результат развёртки, переиспользуя ранее посчитанный.

    Развёртка стоит одного полного расчёта системы на точку сетки, поэтому
    одинаковые развёртки разных метрик считаются один раз.

    Args:
        config (ComputationConfig): Конфигурация вычислений.
        params (SystemParams): Базовые параметры системы.
        parameter (SweepParameter): Описание параметра и сетки значений.
        metric_keys (list[str]): Ключи метрик, которые нужно собрать.
        time_count (int): Число временных точек расчёта.
        cache (dict): Словарь ранее посчитанных развёрток.

    Returns:
        dict[str, NDArray[np.float64]]: Значения метрик формы
            [число точек сетки, time_count].
    """
    cached = cache.get(parameter.cache_key)

    if cached is not None and all(key in cached for key in metric_keys):
        return cached

    with st.spinner(f"⏳ Расчёт развёртки по {parameter.axis_title}..."):
        surfaces = run_parameter_sweep(
            config=config,
            params=params,
            parameter=parameter,
            metric_keys=sorted({*metric_keys, *(cached or {})}),
            time_count=time_count,
        )

    cache[parameter.cache_key] = surfaces
    return surfaces
