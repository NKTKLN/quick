"""Страница для моделирования СМО с нетерпеливыми заявками.

Позволяет пользователю задавать параметры системы, выбирать тип СМО, производить расчет
вероятностей и визуализировать результаты.
"""

from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st
from numpy.typing import NDArray

from app.domain import CalculationMode, ComputationConfig, SystemMode, SystemType
from app.domain.models import (
    BaseSystemParams,
    CalculationSettings,
    MAPSystemParams,
    ServerParams,
    TransientSystemParams,
)
from app.services.systems import system_factory
from pages.components import (
    get_intensity_parameters,
    get_system_mode_type,
    map_intensity_matrix,
    plot_metric,
    plot_probabilities,
    render_calculation_config,
    render_initial_conditions,
    render_time_settings,
    system_capacity_inputs,
)

PLOT_SETTINGS = {
    "throughput": {
        "title_text": "Пропускная способность системы",
        "yaxis_title": "Заявки/ед. времени",
        "line_colors": ["#1f77b4"],
    },
    "avg_buffer_length": {
        "title_text": "Среднее число заявок в буфере",
        "yaxis_title": "Каналы",
        "line_colors": ["#ff7f0e"],
    },
    "avg_system_length": {
        "title_text": "Среднее число заявок в системе",
        "yaxis_title": "Заявки",
        "line_colors": ["#9467bd"],
    },
    "absolute_throughput": {
        "title_text": "Абсолютная пропускная способность",
        "yaxis_title": "Заявки/ед. времени",
        "line_colors": ["#2ca02c"],
    },
    "relative_throughput": {
        "title_text": "Относительная пропускная способность",
        "yaxis_title": "Доля от входящего потока",
        "line_colors": ["#d62728"],
    },
    "rejection_probability": {
        "title_text": "Вероятность отказа",
        "yaxis_title": "Вероятность",
        "line_colors": ["#8c564b"],
    },
    "service_probability": {
        "title_text": "Вероятность обслуживания заявок",
        "yaxis_title": "Вероятность",
        "line_colors": ["#e377c2"],
    },
}


# pylint: disable=too-many-locals
def get_user_inputs() -> tuple[ComputationConfig, ServerParams]:
    """Собирает все входные параметры от пользователя через UI.

    Returns:
        tuple:
            - config (ComputationConfig): Конфигурация для выполнения вычислений.
            - params (ServerParams): Параметры выбранноСМО.
    """
    config = render_calculation_config()
    st.markdown("---")
    system_type, calculation_mode, system_mode = get_system_mode_type()

    st.subheader("⚙️ Параметры системы")
    lambda_rate, mu_rate, nu_rate = get_intensity_parameters(system_type)

    p_rate, q_rate = None, None
    if system_type == SystemType.MAP:
        max_customers, processor_count = system_capacity_inputs(
            system_type, default_max_customers=3
        )
        st.markdown("---")
        p_rate, q_rate = map_intensity_matrix(max_customers)
    else:
        max_customers, processor_count = system_capacity_inputs(system_type)

    count = max_customers

    if system_type == SystemType.MULTI and processor_count is not None:
        count += processor_count + 1
    elif system_type == SystemType.MAP:
        count **= 2

    transient_params = None
    if system_mode == SystemMode.TRANSIENT:
        st.markdown("---")
        state_variables, initial_probabilities = render_initial_conditions(count)

        st.markdown("---")
        time_array = render_time_settings()

        transient_params = TransientSystemParams(
            time_array=time_array,
            state_variables=state_variables,
            initial_probabilities=initial_probabilities,
        )

    base_params = BaseSystemParams(
        mu_rate=mu_rate,
        nu_rate=nu_rate,
        lambda_rate=lambda_rate,
        max_customers=max_customers,
        processor_count=processor_count,
    )

    settings = CalculationSettings(
        calculation_mode=calculation_mode,
        system_mode=system_mode,
        system_type=system_type,
    )

    if system_type == SystemType.MAP and p_rate is not None and q_rate is not None:
        base_params = MAPSystemParams(
            **asdict(base_params),
            p_rate=p_rate.astype(np.float64),
            q_rate=q_rate.astype(np.float64),
        )

    params = ServerParams(
        base_params=base_params,
        transient_params=transient_params,
        settings=settings,
    )
    return config, params


def format_dataframe(
    df: pd.DataFrame, precision: int = 16
) -> pd.io.formats.style.Styler:
    """Форматирует все числовые значения в DataFrame.

    Args:
        df (pd.DataFrame): Исходный DataFrame.
        precision (int): Количество знаков после запятой для форматирования чисел.

    Returns:
        pd.io.formats.style.Styler: Отформатированный Styler для отображения
            в Streamlit.
    """
    return df.style.format(
        lambda x: f"{x:.{precision}f}" if isinstance(x, float) else x
    )


def split_results(
    results_dict: dict[str, NDArray[np.float64]],
) -> tuple[dict[str, NDArray[np.float64]], list]:
    """Разделяет словарь результатов системы на две переменных.

    Разделяет результаты на две переменных одна это матрица вероятностей, а другая это
    словарь со всеми остальными параметрами.

    Args:
        results_dict (dict[str, NDArray[np.float64]]): Словарь результатов вычисления.

    Returns:
        tuple: Кортеж из двух словарей:
            - probability: матрица вероятностей,
            - metrics_dict: словарь со всеми остальными ключами и значениями из
                исходного словаря, кроме "probability".
    """
    results = {k: v[-1] for k, v in results_dict.items() if k != "probability"}
    probabilities = results_dict["probability"][:, 0].tolist()
    return results, probabilities


# ruff: noqa: C901
def main() -> None:
    """Основная функция страницы: UI, вычисление, визуализация."""
    st.title("🧪 Моделирование СМО с нетерпеливыми заявками")
    st.markdown("---")

    try:
        config, params = get_user_inputs()
    except ValueError as e:
        st.error(f"❌ Ошибка в вводных данных: {e}")
        st.stop()

    if st.button("🚀 Применить параметры"):
        try:
            config.validate()
            params.validate()
        except ValueError as e:
            st.error(f"❌ Ошибка в параметрах: {e}")
            st.stop()

        steady_system = system_factory(
            system_mode=SystemMode.STEADY,
            system_type=params.settings.system_type,
            params=params,
            config=config,
        )
        transient_system = None
        if params.settings.system_mode == SystemMode.TRANSIENT:
            transient_system = system_factory(
                system_mode=SystemMode.TRANSIENT,
                system_type=params.settings.system_type,
                params=params,
                config=config,
            )

        try:
            with st.spinner("⏳ Идёт расчёт значений..."):
                steady_system.calculate_probabilities()
                if transient_system is not None:
                    transient_system.calculate_probabilities()
        except ValueError as e:
            st.error(f"❌ Ошибка при вычислении: {e}")
            st.stop()

        st.markdown("---")

        steady_system_results, steady_system_probabilities = split_results(
            steady_system.calculate()
        )
        results = {SystemMode.STEADY.value: steady_system_results}
        probabilities = {SystemMode.STEADY.value: steady_system_probabilities}
        if transient_system is not None:
            st.subheader("📈 Визуализация динамики состояний системы")
            (
                results[SystemMode.TRANSIENT.value],
                probabilities[SystemMode.TRANSIENT.value],
            ) = split_results(transient_system.calculate())

            st.plotly_chart(
                plot_probabilities(
                    transient_system.probabilities,
                    params.transient_params.time_array,
                    params.settings.calculation_mode.value,
                ),
                use_container_width=True,
            )

            for key, value in results[SystemMode.TRANSIENT.value]:
                try:
                    st.plotly_chart(
                        plot_metric(
                            value,
                            params.transient_params.time_array,
                            **PLOT_SETTINGS[key],
                        ),
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"❌ Ошибка в результатах: {e}")

        st.subheader("📊 Вероятности стационарных состояний")
        st.dataframe(format_dataframe(pd.DataFrame(probabilities)))

        if params.settings.calculation_mode != CalculationMode.PROBABILITY:
            st.subheader("📈 Основные характеристики системы")
            st.dataframe(format_dataframe(pd.DataFrame(results)))


main()
