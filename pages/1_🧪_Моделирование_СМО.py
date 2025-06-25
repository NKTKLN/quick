"""Страница для моделирования СМО с нетерпеливыми заявками.

Позволяет пользователю задавать параметры системы, выбирать тип СМО,
производить расчет вероятностей и визуализировать результаты.
"""

import numpy as np
import streamlit as st
from numpy.typing import NDArray

from app.domain import CalculationMode, ComputationConfig, SystemType
from app.domain.models import BasicServerParams, model_factory
from app.services.systems import system_factory
from pages.components import (
    get_intensity_parameters,
    get_system_mode_type,
    map_intensity_matrix,
    plot_probabilities,
    plot_throughput,
    render_calculation_config,
    render_initial_conditions,
    render_time_settings,
    system_capacity_inputs,
)


# pylint: disable=too-many-locals
def get_user_inputs() -> tuple[
    ComputationConfig, BasicServerParams, SystemType, CalculationMode
]:
    """Собирает все входные параметры от пользователя через UI.

    Returns:
        tuple:
            - config: Конфигурация для выполнения вычислений.
            - params: Параметры выбранной системы массового обслуживания (СМО),
                включая структуру, интенсивности, параметры обслуживания и т.д.
            - system_type (SystemType): Тип модели СМО, выбранный пользователем.
            - calculation_mode (str): Режим вычислений, выбранный пользователем.
    """
    config = render_calculation_config()
    st.markdown("---")
    system_type, calculation_mode = get_system_mode_type()

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

    st.markdown("---")
    state_variables, initial_probabilities = render_initial_conditions(count)

    st.markdown("---")
    time_array = render_time_settings()

    base_params: dict[str, int | NDArray] = dict(
        mu_rate=mu_rate,
        max_customers=max_customers,
        time_array=time_array,
        state_variables=state_variables,
        initial_probabilities=initial_probabilities,
        nu_rate=nu_rate,
        lambda_rate=lambda_rate,
    )

    if system_type == SystemType.MULTI and processor_count is not None:
        base_params.update(processor_count=processor_count)
    elif system_type == SystemType.MAP and p_rate is not None and q_rate is not None:
        base_params.update(
            p_rate=p_rate.astype(np.float64),
            q_rate=q_rate.astype(np.float64),
        )

    params = model_factory(system_type, **base_params)
    return config, params, system_type, calculation_mode


def main() -> None:
    """Основная функция страницы: UI, вычисление, визуализация."""
    st.title("🧪 Моделирование СМО с нетерпеливыми заявками")
    st.markdown("---")

    try:
        config, params, system_type, calculation_mode = get_user_inputs()
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

        try:
            system = system_factory(
                system_type,
                calculation_mode,
                params=params,
                config=config,
            )
        except ValueError:
            st.error("❌ Некорректный режим/тип системы")
            st.stop()

        st.success("✅ Параметры успешно заданы!")

        try:
            with st.spinner("⏳ Идёт расчёт значений..."):
                probabilities = system.calculate()
        except ValueError as e:
            st.error(f"❌ Ошибка при вычислении: {e}")
            st.stop()

        st.markdown("---")
        if calculation_mode == CalculationMode.PROBABILITY and isinstance(
            probabilities, np.ndarray
        ):
            st.subheader("📊 Графики вероятностей состояний системы")
            fig = plot_probabilities(probabilities, params.time_array)
        elif calculation_mode == CalculationMode.THROUGHPUT:
            st.subheader("📊 График пропускной способности системы")
            fig = plot_throughput(probabilities, params.time_array)
        else:
            st.error("❌ Некорректный режим/тип системы")
            st.stop()
        st.plotly_chart(fig, use_container_width=True)


main()
