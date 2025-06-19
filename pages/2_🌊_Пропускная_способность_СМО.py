"""Страница для расчёта пропускной способности СМО с нетерпеливыми заявками.

Позволяет задавать параметры системы и вычислительную конфигурацию,
производить расчет пропускной способности и визуализировать результаты.
"""

from dataclasses import asdict

import streamlit as st

from app.domain import (
    BasicSingleServerParams,
    ComputationConfig,
    MergedComputationConfig,
    MpmathComputationConfig,
    MultiServerThroughputParams,
    SingleServerThroughputParams,
)
from app.services import MultiServerThroughputSystem, SingleServerThroughputSystem
from pages.components import (
    calculation_config,
    plot_throughput,
    render_initial_probabilities,
    render_state_variables,
    system_capacity_inputs,
    throughput_intensity_parameters,
    time_settings,
)


def render_description() -> None:
    """Отображает описание модели и основных параметров системы."""
    st.markdown(
        """## 🔍 Описание
На этой странице реализован расчёт пропускной способности
системы массового обслуживания типа **M/M/m/n** с учётом нетерпеливых заявок.

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
    SingleServerThroughputParams | MultiServerThroughputParams,
    str,
]:
    """Собирает все входные параметры от пользователя через UI.

    Returns:
        tuple: Кортеж с параметрами:
            - config: Конфигурация вычислений.
            - params: Параметры СМО.
            - system_type (str): Тип системы — "Однолинейная", "Многолинейная".
    """
    config = calculation_config()
    st.markdown("---")

    st.subheader("🔬 Тип системы")
    system_type = st.selectbox("Выберите тип СМО:", ["Однолинейная", "Многолинейная"])

    st.subheader("⚙️ Параметры системы")
    lambda_rate, mu_rate, nu_rate = throughput_intensity_parameters()
    max_customers, processor_count = system_capacity_inputs(system_type)

    count = max_customers
    if system_type == "Многолинейная":
        count += processor_count + 1

    st.markdown("---")
    state_variables = render_state_variables(count)
    initial_probabilities = render_initial_probabilities(count)

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
                SingleServerThroughputParams(**base_params, nu_rate=nu_rate),
                system_type,
            )
        case "Многолинейная":
            return (
                config,
                MultiServerThroughputParams(
                    **base_params, nu_rate=nu_rate, processor_count=processor_count
                ),
                system_type,
            )
        case _:
            raise


def main() -> None:
    """Основная функция страницы: UI, вычисление, визуализация."""
    st.title("🌊 Пропускная способность СМО с нетерпеливыми заявками")
    render_description()
    st.markdown("---")

    (
        config,
        params,
        system_type,
    ) = get_user_inputs()

    if st.button("🚀 Применить параметры"):
        try:
            params.validate()
        except Exception as e:
            st.error(f"❌ Ошибка в параметрах: {e}")
            st.stop()

        system: SingleServerThroughputSystem | MultiServerThroughputSystem
        match system_type:
            case "Однолинейная":
                system = SingleServerThroughputSystem(params, config)
            case "Многолинейная":
                system = MultiServerThroughputSystem(params, config)
            case _:
                st.error("❌ Некорректный тип системы")
                st.stop()
        st.success("✅ Параметры успешно заданы!")

        try:
            with st.spinner("⏳ Идёт расчёт пропускной способности..."):
                probabilities = system.calculate()
        except Exception as e:
            st.error(f"❌ Ошибка при вычислении: {e}")
            st.stop()

        st.markdown("---")
        st.subheader("📊 График пропускной способности системы")
        fig = plot_throughput(probabilities, params.time_array)
        st.plotly_chart(fig, use_container_width=True)


main()
