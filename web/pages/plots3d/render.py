"""Модуль отображения трёхмерных графиков характеристик СМО.

Модуль отвечает за:
    - расчёт характеристик системы для заданных пользователем параметров;
    - выбор набора метрик, для которых строятся трёхмерные графики;
    - выбор вида третьей оси отдельно для каждого графика;
    - отрисовку поверхностей и фазовых портретов.

Поддерживаются четыре вида третьей оси:
    - отношение ν/μ на отрезке [0, 1] — влияние нетерпеливости заявок;
    - развёртка по параметру (μ, ν, λ) — требует пересчёта системы
      в каждой точке сетки;
    - вторая метрика — фазовый портрет по одному расчёту;
    - номер датчика — покомпонентный расчёт MAP-системы.

Основная функция:
    render_page() — отображает страницу целиком.
"""

from typing import Any

import numpy as np
import streamlit as st
from numpy.typing import NDArray
from pages.components import plot_phase_portrait, plot_sensor_surface, plot_surface
from pages.simulation import PLOT_SETTINGS, get_user_inputs, prepare_system

from quick.domain import CalculationEngine, SystemMode, SystemType
from quick.domain.params import SystemParams

from .sweep import get_sweep, render_nu_mu_ratio_controls, render_parameter_controls

RESULTS_KEY = "plots3d_results"
SENSOR_RESULTS_KEY = "plots3d_sensor_results"
SWEEP_CACHE_KEY = "plots3d_sweep_cache"

AXIS_NU_MU = "Отношение ν/μ"
AXIS_PARAMETER = "Развёртка по параметру"
AXIS_METRIC = "Вторая метрика"
AXIS_SENSOR = "Номер датчика"

DEFAULT_METRIC = "loss_probability"
EXCLUDED_METRICS = ("probability",)
CONTOURS_KEY = "plots3d_show_contours"

# Основные характеристики, для которых строятся графики по умолчанию:
# они полностью описывают качество обслуживания и загрузку системы.
MAIN_METRICS = (
    "loss_probability",
    "rejection_probability",
    "quit_probability",
    "service_probability",
    "throughput",
    "avg_system_length",
    "stability_coefficient",
)

PAGE_GUIDE = """
### 📖 Как читать графики

По оси **X всегда отложено время**, поэтому страница работает только в
переходном режиме. По оси **Z** — значение выбранной характеристики. Смысл
третьей оси (**Y**) задаётся для каждого графика отдельно.

**Отношение ν/μ.** Основной вид третьей оси. Интенсивность обслуживания μ
остаётся неизменной, а интенсивность ухода нетерпеливых заявок задаётся как
ν = (ν/μ)·μ для каждого значения отрезка [0, 1]. Крайние точки отрезка имеют
прямой смысл: **ν/μ = 0** — ухода нет, заявка дожидается обслуживания;
**ν/μ = 1** — заявка покидает очередь в среднем за то же время, за которое
обслуживается. Поверхность показывает, как нетерпеливость заявок меняет
характеристику во времени.

**Развёртка по параметру.** Система полностью пересчитывается для каждого
значения из заданной сетки (например, 12 значений ν от 50 до 150), и
полученные кривые складываются в поверхность. Сечение вдоль времени при
фиксированном Y — это обычный двумерный график со страницы моделирования.
Сечение поперёк, при фиксированном моменте времени, — зависимость
характеристики от параметра в этот момент. Это единственный вид третьей оси,
требующий пересчётов: 12 точек сетки означают 12 решений системы.

**Вторая метрика.** Поверхности здесь нет: строится траектория системы —
одна кривая, у которой по осям X и Y отложены две характеристики, а по оси Z
и цветом — время. Отмечены точки «старт» и «финиш». Сжатие траектории в одну
область означает выход системы на установившийся режим.

**Номер датчика.** Ось Y дискретна: данные существуют только при целых
значениях 1, 2, 3, … Поверхность между ними — интерполяция для наглядности,
физического смысла она не несёт, поэтому сравнивать датчики нужно по
отдельным линиям сечений. Этот вид оси показывает вклад всех датчиков
сразу; чтобы считать характеристики только по состояниям одного датчика,
выберите его в блоке «Режим расчёта по датчикам» и пересчитайте систему.

### 🔤 Обозначения характеристик потоков

| Характеристика | Смысл |
| --- | --- |
| λ(t) | Текущая интенсивность входного потока. Меняется во времени, потому что меняется распределение фаз MAP-процесса |
| v_serv(t) | Фактический поток обслуженных заявок: μ, взвешенная на вероятность занятости прибора |
| v_loss(t) | Фактический поток ушедших нетерпеливых заявок: ν, взвешенная на заполнение буфера |
| α(t) | Распределение фаз MAP: вклад каждого датчика во входной поток в данный момент |
"""


def _metric_title(metric_key: str) -> str:
    """Возвращает заголовок метрики.

    Args:
        metric_key (str): Ключ метрики.

    Returns:
        str: Человекочитаемое название метрики.
    """
    return str(PLOT_SETTINGS.get(metric_key, {}).get("title_text", metric_key))


def _metric_units(metric_key: str) -> str:
    """Возвращает подпись оси значений метрики.

    Args:
        metric_key (str): Ключ метрики.

    Returns:
        str: Единицы измерения метрики.
    """
    return str(PLOT_SETTINGS.get(metric_key, {}).get("yaxis_title", "Value"))


def _available_metrics(results: dict[str, NDArray[np.float64]]) -> list[str]:
    """Возвращает метрики, пригодные для трёхмерной визуализации.

    Args:
        results (dict[str, NDArray[np.float64]]): Результаты расчёта.

    Returns:
        list[str]: Ключи метрик.
    """
    return [key for key in results if key not in EXCLUDED_METRICS]


def _time_series_metrics(
    results: dict[str, NDArray[np.float64]], time_count: int
) -> list[str]:
    """Возвращает метрики, представленные одной кривой во времени.

    Args:
        results (dict[str, NDArray[np.float64]]): Результаты расчёта.
        time_count (int): Число временных точек.

    Returns:
        list[str]: Ключи одномерных по времени метрик.
    """
    metrics = []

    for key in _available_metrics(results):
        values = np.asarray(results[key])
        if values.ndim == 1 and values.shape[0] == time_count:
            metrics.append(key)

    return metrics


def compute_results(config: Any, params: SystemParams) -> None:
    """Рассчитывает характеристики системы и сохраняет их в состоянии сессии.

    Args:
        config (ComputationConfig): Конфигурация вычислений.
        params (SystemParams): Параметры системы.
    """
    system = prepare_system(config, params)

    st.session_state[RESULTS_KEY] = {
        "config": config,
        "params": params,
        "results": system.calculate(),
    }
    st.session_state[SWEEP_CACHE_KEY] = {}
    st.session_state.pop(SENSOR_RESULTS_KEY, None)


def _sensor_results(state: dict) -> dict[str, NDArray[np.float64]]:
    """Возвращает характеристики, рассчитанные отдельно для каждого датчика.

    Расчёт выполняется лениво: только когда хотя бы один график использует
    третью ось по датчикам.

    Args:
        state (dict): Сохранённое состояние страницы.

    Returns:
        dict[str, NDArray[np.float64]]: Результаты покомпонентного расчёта.
    """
    if SENSOR_RESULTS_KEY not in st.session_state:
        with st.spinner("⏳ Расчёт характеристик по каждому датчику..."):
            system = prepare_system(state["config"], state["params"], per_sensor=True)
            st.session_state[SENSOR_RESULTS_KEY] = system.calculate()

    return dict(st.session_state[SENSOR_RESULTS_KEY])


def _axis_options(params: SystemParams) -> list[str]:
    """Возвращает виды третьей оси, доступные для текущей системы.

    Args:
        params (SystemParams): Параметры системы.

    Returns:
        list[str]: Названия видов третьей оси.
    """
    options = [AXIS_NU_MU, AXIS_PARAMETER, AXIS_METRIC]

    if params.calculation_params.system_type == SystemType.MAP:
        options.append(AXIS_SENSOR)

    return options


def _show_contours() -> bool:
    """Возвращает, наносить ли линии уровня на поверхности.

    Returns:
        bool: Значение переключателя контуров.
    """
    return bool(st.session_state.get(CONTOURS_KEY, True))


def _sweep_surface(state: dict, metric_key: str, parameter: Any) -> Any:
    """Строит поверхность «время × параметр × метрика» по готовой развёртке.

    Args:
        state (dict): Сохранённое состояние страницы.
        metric_key (str): Ключ отображаемой метрики.
        parameter (SweepParameter): Описание развёртки и сетки её значений.

    Returns:
        go.Figure: Поверхность Plotly.
    """
    time_array = state["params"].transient_params.time_array

    surfaces = get_sweep(
        config=state["config"],
        params=state["params"],
        parameter=parameter,
        metric_keys=[metric_key],
        time_count=len(time_array),
        cache=st.session_state[SWEEP_CACHE_KEY],
    )

    return plot_surface(
        values=surfaces[metric_key],
        time_array=time_array,
        axis_values=parameter.grid,
        title_text=f"{_metric_title(metric_key)}: развёртка по {parameter.axis_title}",
        yaxis_title=parameter.axis_title,
        zaxis_title=_metric_units(metric_key),
        show_contours=_show_contours(),
    )


def _supports_sweep(state: dict, metric_key: str) -> bool:
    """Проверяет, можно ли развернуть метрику по параметру.

    Args:
        state (dict): Сохранённое состояние страницы.
        metric_key (str): Ключ метрики.

    Returns:
        bool: True, если метрика представлена одной кривой во времени.
    """
    time_count = len(state["params"].transient_params.time_array)

    if metric_key in _time_series_metrics(state["results"], time_count):
        return True

    st.info(
        "ℹ️ Развёртка доступна только для метрик, представленных "
        "одной кривой во времени. Для метрик, уже разложенных по датчикам, "
        "выберите третью ось «Номер датчика»."
    )
    return False


def _render_nu_mu_axis(state: dict, metric_key: str) -> Any:
    """Строит поверхность «время × ν/μ × метрика».

    Отношение ν/μ пробегает отрезок [0, 1]: от полного отсутствия ухода
    нетерпеливых заявок до случая ν = μ. Интенсивность обслуживания при
    этом не меняется, поэтому поверхность показывает вклад именно
    нетерпеливости заявок.

    Args:
        state (dict): Сохранённое состояние страницы.
        metric_key (str): Ключ отображаемой метрики.

    Returns:
        go.Figure | None: График или None, если метрика не поддерживается.
    """
    if not _supports_sweep(state, metric_key):
        return None

    parameter = render_nu_mu_ratio_controls(key_prefix=f"{metric_key}_nu_mu")

    return _sweep_surface(state, metric_key, parameter)


def _render_parameter_axis(state: dict, metric_key: str) -> Any:
    """Строит поверхность «время × параметр × метрика».

    Args:
        state (dict): Сохранённое состояние страницы.
        metric_key (str): Ключ отображаемой метрики.

    Returns:
        go.Figure | None: График или None, если метрика не поддерживается.
    """
    if not _supports_sweep(state, metric_key):
        return None

    parameter = render_parameter_controls(
        state["params"], key_prefix=f"{metric_key}_sweep"
    )

    return _sweep_surface(state, metric_key, parameter)


def _render_metric_axis(state: dict, metric_key: str) -> Any:
    """Строит фазовый портрет «метрика × метрика × время».

    Args:
        state (dict): Сохранённое состояние страницы.
        metric_key (str): Ключ отображаемой метрики.

    Returns:
        go.Figure | None: График или None, если метрика не поддерживается.
    """
    results = state["results"]
    time_array = state["params"].transient_params.time_array
    metrics = _time_series_metrics(results, len(time_array))

    if metric_key not in metrics:
        st.info(
            "ℹ️ Фазовый портрет доступен только для метрик, представленных "
            "одной кривой во времени."
        )
        return None

    other_metrics = [key for key in metrics if key != metric_key]

    if not other_metrics:
        st.info("ℹ️ Нет второй метрики для построения фазового портрета.")
        return None

    other_key = st.selectbox(
        "Вторая метрика (ось Y)",
        other_metrics,
        format_func=_metric_title,
        key=f"{metric_key}_second_metric",
    )

    return plot_phase_portrait(
        x_values=np.asarray(results[metric_key], dtype=np.float64),
        y_values=np.asarray(results[other_key], dtype=np.float64),
        time_array=time_array,
        title_text=f"{_metric_title(metric_key)} × {_metric_title(other_key)}",
        xaxis_title=_metric_title(metric_key),
        yaxis_title=_metric_title(other_key),
    )


def _render_sensor_axis(state: dict, metric_key: str) -> Any:
    """Строит поверхность «время × номер датчика × метрика».

    Args:
        state (dict): Сохранённое состояние страницы.
        metric_key (str): Ключ отображаемой метрики.

    Returns:
        go.Figure | None: График или None, если метрика не раскладывается
            по датчикам.
    """
    time_array = state["params"].transient_params.time_array
    values = np.asarray(_sensor_results(state).get(metric_key), dtype=np.float64)

    if values.ndim != 2 or values.shape[0] != len(time_array):
        st.info(
            "ℹ️ Эта характеристика описывает систему целиком и не "
            "раскладывается по датчикам."
        )
        return None

    return plot_sensor_surface(
        values=values,
        time_array=time_array,
        title_text=f"{_metric_title(metric_key)} по датчикам",
        zaxis_title=_metric_units(metric_key),
        show_contours=_show_contours(),
    )


def _render_metric_block(state: dict, metric_key: str) -> None:
    """Отображает один график вместе с настройкой его третьей оси.

    Args:
        state (dict): Сохранённое состояние страницы.
        metric_key (str): Ключ отображаемой метрики.
    """
    options = _axis_options(state["params"])
    time_count = len(state["params"].transient_params.time_array)

    # Метрики, уже разложенные по датчикам, не являются одной кривой во
    # времени: для них ось по датчикам — единственный подходящий вариант.
    is_time_series = metric_key in _time_series_metrics(state["results"], time_count)
    default_index = (
        options.index(AXIS_SENSOR)
        if not is_time_series and AXIS_SENSOR in options
        else 0
    )

    with st.expander(f"📈 {_metric_title(metric_key)}", expanded=True):
        axis_kind = st.selectbox(
            "Третья ось",
            options,
            index=default_index,
            key=f"{metric_key}_axis_kind",
        )

        renderers = {
            AXIS_NU_MU: _render_nu_mu_axis,
            AXIS_PARAMETER: _render_parameter_axis,
            AXIS_METRIC: _render_metric_axis,
            AXIS_SENSOR: _render_sensor_axis,
        }

        try:
            figure = renderers[axis_kind](state, metric_key)
        except (ValueError, TypeError, KeyError) as e:
            st.error(f"❌ Не удалось построить график: {e}")
            return

        if figure is not None:
            st.plotly_chart(figure, width="stretch")


def _render_results(state: dict) -> None:
    """Отображает выбор метрик и все построенные графики.

    Args:
        state (dict): Сохранённое состояние страницы.
    """
    metrics = _available_metrics(state["results"])
    default_metrics = [key for key in MAIN_METRICS if key in metrics]

    if not default_metrics:
        default_metrics = [DEFAULT_METRIC] if DEFAULT_METRIC in metrics else metrics[:1]

    st.subheader("🧊 Трёхмерная визуализация характеристик")
    st.caption(
        "Для каждой метрики третья ось выбирается отдельно. Развёртка по "
        "ν/μ и по параметру требует полного пересчёта системы в каждой точке "
        "сетки; одинаковые развёртки разных графиков считаются один раз."
    )

    selected_metrics = st.multiselect(
        "Метрики для построения",
        metrics,
        default=default_metrics,
        format_func=_metric_title,
        key="plots3d_selected_metrics",
    )

    st.checkbox(
        "Контурные линии на поверхности",
        value=True,
        key=CONTOURS_KEY,
        help=(
            "Линии уровня показывают области локальных максимумов, минимумов "
            "и резких изменений характеристики."
        ),
    )

    if not selected_metrics:
        st.info("ℹ️ Выберите хотя бы одну метрику.")
        return

    for metric_key in selected_metrics:
        _render_metric_block(state, metric_key)


def render_page() -> None:
    """Отображает страницу трёхмерной визуализации целиком.

    Стационарный режим на этой странице недоступен: все графики строятся по
    оси времени, которая существует только в переходном режиме.
    """
    st.markdown(PAGE_GUIDE)
    st.markdown("---")

    config, params, _ = get_user_inputs(
        system_modes=[SystemMode.TRANSIENT],
        default_engine=CalculationEngine.NUMPY,
    )

    if st.button("🚀 Рассчитать характеристики"):
        compute_results(config, params)

    state = st.session_state.get(RESULTS_KEY)

    if state is None:
        return

    st.markdown("---")
    _render_results(state)
