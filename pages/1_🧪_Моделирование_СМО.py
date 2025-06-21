"""Страница для моделирования СМО с нетерпеливыми заявками.

Позволяет пользователю задавать параметры системы, выбирать тип СМО,
производить расчет вероятностей и визуализировать результаты.
"""

from dataclasses import asdict

import numpy as np
import streamlit as st

from app.domain import (
    BasicSingleServerParams,
    ComputationConfig,
    MAPServerParams,
    MergedComputationConfig,
    MpmathComputationConfig,
    MultiServerParams,
    SingleServerParams,
)
from app.services import MAPServerSystem, MultiServerSystem, SingleServerSystem
from pages.components import (
    calculation_config,
    intensity_parameters,
    map_intensity_matrix,
    map_intensity_parameters,
    plot_probabilities,
    render_initial_conditions,
    system_capacity_inputs,
    time_settings,
)


def render_description() -> None:
    """Отображает описание модели и основных параметров системы."""
    st.markdown(
        """## 🔍 Описание
На этой странице реализовано численное моделирование СМО типа **M/M/m/n с нетерпеливыми\
 заявками**, применяемой в медицинских информационно-измерительных системах.

## 📌 Основные параметры:
- **λ (лямбда)** — интенсивность поступления заявок (пакетов/с)
- **μ (мю)** — интенсивность обслуживания (пакетов/с)
- **ν (ню)** — интенсивность ухода нетерпеливых заявок (пакетов/с)
- **n** — размер буфера (максимальное количество заявок в системе)
- **m** — количество обслуживающих процессоров (только для многолинейной системы)
    """
    )


def get_user_inputs() -> tuple[
    ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    SingleServerParams | MultiServerParams | MAPServerParams,
    str,
]:
    """Собирает все входные параметры от пользователя через UI.

    Returns:
        tuple: Кортеж с параметрами:
            - config: Конфигурация вычислений.
            - params: Параметры СМО.
            - system_type (str): Тип системы — "Однолинейная", "Многолинейная"
                или "С MAP-потоками".
    """
    config = calculation_config()
    st.markdown("---")

    st.subheader("🔬 Тип системы")
    system_type = st.selectbox(
        "Выберите тип СМО:", ["Однолинейная", "Многолинейная", "С MAP-потоками"]
    )

    st.subheader("⚙️ Параметры системы")
    if system_type == "С MAP-потоками":
        lambda_rate, mu_rate, nu_rate = map_intensity_parameters()
        max_customers, processor_count = system_capacity_inputs(
            system_type, default_max_customers=3
        )
        st.markdown("---")
        p_rate, q_rate = map_intensity_matrix(max_customers)
    else:
        lambda_rate, mu_rate, nu_rate = intensity_parameters()
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

    base_params = asdict(
        BasicSingleServerParams(
            lambda_rate=lambda_rate,
            mu_rate=mu_rate,
            max_customers=max_customers,
            time_array=time_array,
            state_variables=state_variables,
            initial_probabilities=initial_probabilities,
        )
    )

    match system_type:
        case "Однолинейная":
            return (
                config,
                SingleServerParams(**base_params, nu_rate=nu_rate),
                system_type,
            )
        case "Многолинейная":
            return (
                config,
                MultiServerParams(
                    **base_params, nu_rate=nu_rate, processor_count=processor_count
                ),
                system_type,
            )
        case "С MAP-потоками":
            return (
                config,
                MAPServerParams(
                    **base_params,
                    nu_rate=nu_rate,
                    p_rate=p_rate.astype(dtype=np.float64),
                    q_rate=q_rate.astype(dtype=np.float64),
                ),
                system_type,
            )
        case _:
            raise


def main() -> None:
    """Основная функция страницы: UI, вычисление, визуализация."""
    st.title("🧪 Моделирование СМО с нетерпеливыми заявками")
    render_description()
    st.markdown("---")

    config, params, system_type = get_user_inputs()

    if st.button("🚀 Применить параметры"):
        try:
            config.validate()
            params.validate()
        except Exception as e:
            st.error(f"❌ Ошибка в параметрах: {e}")
            st.stop()

        system: SingleServerSystem | MultiServerSystem | MAPServerSystem
        match system_type:
            case "Однолинейная":
                system = SingleServerSystem(params, config)
            case "Многолинейная":
                system = MultiServerSystem(params, config)
            case "С MAP-потоками":
                system = MAPServerSystem(params, config)
            case _:
                st.error("❌ Некорректный тип системы")
                st.stop()

        st.success("✅ Параметры успешно заданы!")

        try:
            with st.spinner("⏳ Идёт расчёт вероятностей состояния..."):
                probabilities = system.calculate()
        except Exception as e:
            st.error(f"❌ Ошибка при вычислении: {e}")
            st.stop()

        st.markdown("---")
        st.subheader("📊 Графики вероятностей состояний системы")
        fig = plot_probabilities(probabilities, params.time_array)
        st.plotly_chart(fig, use_container_width=True)


main()
