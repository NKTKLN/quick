"""Страница для аналитического моделирования СМО с нетерпеливыми заявками.

Позволяет пользователю задавать параметры системы, выбирать тип СМО,
производить расчет вероятностей и визуализировать результаты.
"""

import numpy as np
import streamlit as st

from app.domain import (
    MAPServerParams,
    MAPServerThroughputParams,
    MultiServerParams,
    MultiServerThroughputParams,
    SingleServerParams,
    SingleServerThroughputParams,
)
from app.services import (
    MAPServerSystem,
    MAPServerThroughputSystem,
    MultiServerSystem,
    MultiServerThroughputSystem,
    SingleServerSystem,
    SingleServerThroughputSystem,
)
from pages.components import (
    calculation_config,
    get_intensity_parameters,
    get_system_mode_type,
    map_intensity_matrix,
    plot_probabilities,
    plot_throughput,
    render_initial_conditions,
    system_capacity_inputs,
    time_settings,
)


def get_user_inputs() -> tuple:
    """Собирает все входные параметры от пользователя через UI.

    Returns:
        tuple:
            - config: Конфигурация для выполнения вычислений.
            - params: Параметры выбранной системы массового обслуживания (СМО),
                включая структуру, интенсивности, параметры обслуживания и т.д.
            - system_type (str): Тип модели СМО, выбранный пользователем.
            - calculation_mode (str): Режим вычислений, выбранный пользователем.
    """
    config = calculation_config()
    st.markdown("---")
    system_type, calculation_mode = get_system_mode_type()

    st.subheader("⚙️ Параметры системы")
    lambda_rate, mu_rate, nu_rate = get_intensity_parameters(
        system_type, calculation_mode
    )

    if system_type == "С MAP-потоками":
        max_customers, processor_count = system_capacity_inputs(
            system_type, default_max_customers=3
        )
        st.markdown("---")
        p_rate, q_rate = map_intensity_matrix(max_customers)
    else:
        max_customers, processor_count = system_capacity_inputs(system_type)

    count = max_customers
    if system_type == "Многолинейная":
        count += processor_count + 1
    elif system_type == "С MAP-потоками":
        count **= 2

    st.markdown("---")
    state_variables, initial_probabilities = render_initial_conditions(count)

    st.markdown("---")
    time_array = time_settings()

    base_params = dict(
        mu_rate=mu_rate,
        max_customers=max_customers,
        time_array=time_array,
        state_variables=state_variables,
        initial_probabilities=initial_probabilities,
    )

    match system_type:
        case "Однолинейная":
            base_params.update(lambda_rate=lambda_rate)
        case "Многолинейная":
            base_params.update(lambda_rate=lambda_rate, processor_count=processor_count)
        case "С MAP-потоками":
            base_params.update(
                lambda_rate=lambda_rate,
                p_rate=p_rate.astype(np.float64),
                q_rate=q_rate.astype(np.float64),
            )
        case _:
            raise ValueError(f"Неподдерживаемый тип системы: {system_type}")

    param_classes = {
        ("Вероятностный", "Однолинейная"): SingleServerParams,
        ("Вероятностный", "Многолинейная"): MultiServerParams,
        ("Вероятностный", "С MAP-потоками"): MAPServerParams,
        ("Пропускная способность", "Однолинейная"): SingleServerThroughputParams,
        ("Пропускная способность", "Многолинейная"): MultiServerThroughputParams,
        ("Пропускная способность", "С MAP-потоками"): MAPServerThroughputParams,
    }

    key = (calculation_mode, system_type)
    ParamClass = param_classes.get(key)
    if not ParamClass:
        raise ValueError(f"Неподдерживаемый режим/тип системы: {key}")

    params = ParamClass(**base_params, nu_rate=nu_rate)
    return config, params, system_type, calculation_mode


def main() -> None:
    """Основная функция страницы: UI, вычисление, визуализация."""
    st.title("🧪 Аналитическое моделирование СМО с нетерпеливыми заявками")
    st.markdown("---")

    try:
        config, params, system_type, calculation_mode = get_user_inputs()
    except Exception as e:
        st.error(f"❌ Ошибка в вводных данных: {e}")
        st.stop()

    system_classes = {
        ("Вероятностный", "Однолинейная"): SingleServerSystem,
        ("Вероятностный", "Многолинейная"): MultiServerSystem,
        ("Вероятностный", "С MAP-потоками"): MAPServerSystem,
        ("Пропускная способность", "Однолинейная"): SingleServerThroughputSystem,
        ("Пропускная способность", "Многолинейная"): MultiServerThroughputSystem,
        ("Пропускная способность", "С MAP-потоками"): MAPServerThroughputSystem,
    }

    if st.button("🚀 Применить параметры"):
        try:
            config.validate()
            params.validate()
        except Exception as e:
            st.error(f"❌ Ошибка в параметрах: {e}")
            st.stop()

        key = (calculation_mode, system_type)
        SystemClass = system_classes.get(key)
        if not SystemClass:
            st.error("❌ Некорректный режим/тип системы")
            st.stop()

        system = SystemClass(params, config)
        st.success("✅ Параметры успешно заданы!")

        # try:
        with st.spinner("⏳ Идёт расчёт значений..."):
            probabilities = system.calculate()
        # except Exception as e:
        #     st.error(f"❌ Ошибка при вычислении: {e}")
        #     st.stop()

        st.markdown("---")
        match calculation_mode:
            case "Вероятностный":
                st.subheader("📊 Графики вероятностей состояний системы")
                fig = plot_probabilities(probabilities, params.time_array)
            case "Пропускная способность":
                st.subheader("📊 График пропускной способности системы")
                fig = plot_throughput(probabilities, params.time_array)
            case _:
                st.error("❌ Некорректный режим/тип системы")
                st.stop()
        st.plotly_chart(fig, use_container_width=True)


main()
