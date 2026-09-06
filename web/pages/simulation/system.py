"""Содержит функции для подготовки системы и отображения результатов моделирования.

Модуль отвечает за:
    - валидацию конфигурации вычислений и параметров системы;
    - создание экземпляра системы для переходного или стационарного режима;
    - запуск предварительного расчёта вероятностей состояний;
    - визуализацию вероятностей и вычисленных метрик для переходного режима;
    - отображение таблицы стационарных состояний для стационарного режима;
    - экспорт результатов моделирования в JSON.

Основные функции:
    prepare_system() — проверяет параметры, создаёт систему и
    выполняет предварительный расчёт вероятностей состояний.

    render_transient_results() — отображает графики результатов моделирования
    для переходного режима и добавляет кнопку скачивания данных в формате JSON.

    render_steady_table() — отображает таблицу стационарных состояний
    системы с индексами состояний и значениями вероятностей без округления.
"""

import numpy as np
import pandas as pd
import streamlit as st
from pages.components import (
    plot_metric,
    plot_probabilities,
    plot_stability_margin,
    render_stability_summary,
    sensor_label,
    stability_reference_lines,
)

from quick.domain import ComputationConfig, StabilityMetric, SystemMode
from quick.domain.params import SystemParams
from quick.services.systems import system_factory
from quick.services.systems.base import BaseServerSystem
from quick.services.systems.steady_state import MultiServerSteadyStateSystem

from .constants import PLOT_SETTINGS
from .serializers import results_to_json


def prepare_system(
    config: ComputationConfig, params: SystemParams, per_sensor: bool = False
):
    """Проверяет параметры, создаёт систему и запускает предварительный расчёт.

    Args:
        config (ComputationConfig): Конфигурация вычислений.
        params (SystemParams): Параметры системы.
        per_sensor (bool): Выполнять расчёт отдельно для каждого сенсора.
            Поддерживается только для систем типа MAP.

    Returns:
        Any | None: Экземпляр переходной системы, если выбран режим TRANSIENT.
        В противном случае — None.
    """
    try:
        config.validate()
        params.validate()
    except ValueError as e:
        st.error(f"❌ Ошибка в параметрах: {e}")
        st.stop()

    system = None
    if params.calculation_params.system_mode == SystemMode.TRANSIENT:
        system = system_factory(
            system_mode=SystemMode.TRANSIENT,
            system_type=params.calculation_params.system_type,
            per_sensor=per_sensor,
            params=params,
            config=config,
        )
    elif params.calculation_params.system_mode == SystemMode.STEADY:
        system = system_factory(
            system_mode=SystemMode.STEADY,
            system_type=params.calculation_params.system_type,
            params=params,
            config=config,
        )

    try:
        with st.spinner("⏳ Идёт расчёт значений..."):
            if system is not None:
                system.calculate_probabilities()
    except ValueError as e:
        st.error(f"❌ Ошибка при вычислении: {e}")
        st.stop()

    return system


def render_transient_results(system: BaseServerSystem, params: SystemParams) -> None:
    """Отображает результаты режима и кнопку экспорта JSON.

    Args:
        system (BaseServerSystem): Объект системы.
        params (SystemParams): Параметры системы.

    Returns:
        None: Функция ничего не возвращает.
    """
    st.subheader("📈 Визуализация динамики состояний системы")
    results = system.calculate()

    _render_stability(system, params)

    st.plotly_chart(
        plot_probabilities(
            system.probabilities,
            params.transient_params.time_array,
            "Вероятности",
        ),
        width="stretch",
    )

    for key, value in results.items():
        try:
            if key in ["probability", "avg_buffer_length_by_sensor"]:
                continue

            # У коэффициента устойчивости единица — граница допустимого,
            # поэтому она наносится на график как опорная линия.
            reference_lines = (
                stability_reference_lines(params.transient_params.time_array)
                if key == "stability_coefficient"
                else None
            )

            st.plotly_chart(
                plot_metric(
                    value,
                    params.transient_params.time_array,
                    reference_lines=reference_lines,
                    **PLOT_SETTINGS[key],
                ),
                width="stretch",
            )
        except Exception as e:
            st.error(f"❌ Ошибка в результатах: {e}")

    st.markdown("---")

    try:
        json_data = results_to_json(results, params.transient_params.time_array)

        st.download_button(
            label="📥 Скачать результаты в JSON",
            data=json_data,
            file_name="simulation_results.json",
            mime="application/json",
            key="download_json_button",
        )
    except Exception as e:
        st.warning(f"⚠️ Не удалось подготовить JSON для скачивания: {e}")
        st.info(
            "Попробуйте экспортировать данные в другом формате или обратитесь к разработчику."
        )


def _render_stability(system: BaseServerSystem, params: SystemParams) -> None:
    """Отображает итоговую оценку устойчивости системы.

    Args:
        system (BaseServerSystem): Объект системы.
        params (SystemParams): Параметры системы.
    """
    st.markdown("### 🛡️ Оценка устойчивости")

    try:
        results = system.evaluate_stability_components()
        base_metric = np.asarray(
            system.calculate_stability_base_metric(),
            dtype=np.float64,
        )
    except (ValueError, TypeError) as e:
        st.warning(f"⚠️ Не удалось оценить устойчивость: {e}")
        return

    if base_metric.ndim == 1:
        base_metric = base_metric[:, np.newaxis]

    if params.stability_params.metric == StabilityMetric.THROUGHPUT:
        yaxis_title = "A(t), заявки/ед. времени"
        base_label = "A(t)"
    else:
        yaxis_title = "a(t) = 1 - P_loss(t)"
        base_label = "a(t)"

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]

    def render_margin_plot(index: int, label: str) -> None:
        """Отображает график запаса и подпись площади под ним."""
        result = results[index]
        st.plotly_chart(
            plot_stability_margin(
                values=base_metric[:, index],
                time_array=params.transient_params.time_array,
                result=result,
                yaxis_title=yaxis_title,
                series_label=label,
                line_color=colors[index % len(colors)],
            ),
            width="stretch",
        )
        st.caption(
            "Площадь закрашенного участка над критическим уровнем: "
            f"S₊ = {result.area_above_critical:.6g}. "
            "Расчёт: S₊ = ∫ max(a(t) − a_кр, 0) dt на переходном интервале."
        )

    # При покомпонентном расчёте критический уровень предъявляется к
    # каждому датчику отдельно, поэтому и оценка выводится по каждому.
    if len(results) == 1:
        render_stability_summary(results[0], params.stability_params)
        render_margin_plot(0, base_label)
        return

    for index, result in enumerate(results):
        label = sensor_label(index)
        st.markdown(f"**{label}**")
        render_stability_summary(result, params.stability_params)
        render_margin_plot(index, f"{base_label} — {label}")


def render_steady_table(system: MultiServerSteadyStateSystem) -> None:
    """Отображает таблицу стационарных состояний системы.

    В таблице:
        - первый столбец: индекс состояния от 0 до n-1;
        - второй столбец: значение вероятности без округления.

    Args:
        system (BaseServerSystem): Объект системы.

    Returns:
        None
    """
    probabilities = getattr(system, "probabilities", None)
    if probabilities is None:
        st.warning("⚠️ Вероятности стационарных состояний отсутствуют.")
        return

    try:
        probabilities_array = np.asarray(probabilities).reshape(-1)

        df = pd.DataFrame(
            {
                "Индекс": np.arange(len(probabilities_array)),
                "Значение": probabilities_array,
            }
        )

        df["Значение"] = df["Значение"].map(lambda x: f"{x:.15e}")

        st.subheader("Стационарные состояния системы")
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception as e:
        st.error(f"❌ Ошибка при отображении стационарных состояний: {e}")
