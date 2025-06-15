"""Страница для моделирования СМО с нетерпеливыми заявками.

Позволяет пользователю задавать параметры системы, выбирать тип СМО,
производить расчет вероятностей и визуализировать результаты.
"""

import numpy as np
import streamlit as st

from app.domain import (
    ComputationConfig,
    MergedComputationConfig,
    MpmathComputationConfig,
    MultiServerParams,
    SingleServerParams,
)
from app.services.probability import MultiServerSystem, SingleServerSystem
from pages.components import (
    calculation_config,
    intensity_parameters,
    plot_probabilities,
    render_initial_probabilities,
    render_state_variables,
    system_capacity_inputs,
    time_settings,
)


def render_description() -> None:
    """Отображает описание модели и основных параметров системы."""
    st.markdown(
        """## 🔍 Описание
На этой странице реализовано численное моделирование СМО типа **M/M/m/n с нетерпеливыми\
    заявками**, применяемой в медицинских информационно-измерительных системах.

## 📌 Основные параметры системы:
- **λ (лямбда)** — интенсивность поступления заявок (пакетов/с)
- **μ (мю)** — интенсивность обслуживания (пакетов/с)
- **ν (ню)** — интенсивность ухода нетерпеливых заявок (пакетов/с)
- **n** — размер буфера (макс. кол-во заявок в системе, включая обслуживаемую)
- **m** (опционально) — количество обслуживающих процессоров
"""
    )


def get_user_inputs() -> tuple[
    ComputationConfig | MpmathComputationConfig | MergedComputationConfig,
    str,
    float,
    float,
    float,
    int,
    int | None,
    np.ndarray[np.float64],
    np.ndarray[np.float64],
    np.ndarray[np.float64],
]:
    """Собирает все входные параметры от пользователя через UI.

    Returns:
        tuple: Параметры конфигурации вычислений, тип системы, интенсивности,
               размеры, временной массив, переменные состояния и начальные вероятности.
    """
    config = calculation_config()
    st.markdown("---")

    st.subheader("🔬 Тип системы")
    system_type = st.selectbox("Выберите тип СМО:", ["Однолинейная", "Многолинейная"])

    st.subheader("⚙️ Параметры системы")
    lambda_rate, mu_rate, nu_rate = intensity_parameters()
    max_customers, processor_count = system_capacity_inputs(system_type)
    time_array = time_settings()

    count = max_customers
    if system_type == "Многолинейная":
        count = max_customers + processor_count

    state_variables = render_state_variables(count)
    initial_probabilities = render_initial_probabilities(count)

    return (
        config,
        system_type,
        lambda_rate,
        mu_rate,
        nu_rate,
        max_customers,
        processor_count,
        time_array,
        state_variables,
        initial_probabilities,
    )


def validate_inputs(
    initial_probabilities: np.ndarray[np.float64],
    state_variables: np.ndarray[np.float64],
    lambda_rate: float,
    mu_rate: float,
    nu_rate: float,
    time_array: np.ndarray[np.float64],
) -> bool:
    """Проверяет корректность введенных пользователем данных.

    Args:
        initial_probabilities (np.ndarray[np.float64]): Массив начальных вероятностей.
        state_variables (np.ndarray[np.float64]): Массив переменных состояния.
        lambda_rate (float): Интенсивность поступления заявок λ.
        mu_rate (float): Интенсивность обслуживания μ.
        nu_rate (float): Интенсивность ухода нетерпеливых заявок ν.
        time_array (np.ndarray[np.float64]): Массив времени для моделирования.

    Returns:
        bool: True, если все проверки пройдены, иначе False.
    """
    if not np.isclose(np.sum(initial_probabilities), 1.0):
        st.error("Сумма начальных вероятностей должна быть равна 1.")
        return False

    if len(state_variables) != len(initial_probabilities):
        st.error(
            "Количество переменных состояния должно совпадать с количеством \
                начальных вероятностей."
        )
        return False

    if np.any(initial_probabilities < 0) or np.any(initial_probabilities > 1):
        st.error("Все вероятности должны быть в диапазоне от 0 до 1.")
        return False

    if np.any(state_variables < 0):
        st.error("Переменные состояния не могут быть отрицательными.")
        return False

    if time_array is None or len(time_array) == 0:
        st.error("Временной диапазон некорректен или пуст.")
        return False

    if lambda_rate <= 0 or mu_rate <= 0 or nu_rate <= 0:
        st.error("Все интенсивности (λ, μ, ν) должны быть положительными.")
        return False

    return True


def main() -> None:
    """Основная функция рендеринга страницы."""
    st.title("🧪 Моделирование СМО с нетерпеливыми заявками")
    render_description()
    st.markdown("---")
    (
        config,
        system_type,
        lambda_rate,
        mu_rate,
        nu_rate,
        max_customers,
        processor_count,
        time_array,
        state_variables,
        initial_probabilities,
    ) = get_user_inputs()

    if st.button("🚀 Применить параметры"):
        if not validate_inputs(
            initial_probabilities,
            state_variables,
            lambda_rate,
            mu_rate,
            nu_rate,
            time_array,
        ):
            st.stop()

        system: SingleServerSystem | MultiServerSystem
        if system_type == "Многолинейная":
            system = MultiServerSystem(
                MultiServerParams(
                    lambda_rate=lambda_rate,
                    mu_rate=mu_rate,
                    nu_rate=nu_rate,
                    max_customers=max_customers,
                    processor_count=processor_count,
                    time_array=time_array,
                    state_variables=state_variables,
                    initial_probabilities=initial_probabilities,
                ),
                config,
            )
        else:
            system = SingleServerSystem(
                SingleServerParams(
                    lambda_rate=lambda_rate,
                    mu_rate=mu_rate,
                    nu_rate=nu_rate,
                    max_customers=max_customers,
                    time_array=time_array,
                    state_variables=state_variables,
                    initial_probabilities=initial_probabilities,
                ),
                config,
            )

        st.success("✅ Параметры успешно заданы!")

        try:
            with st.spinner("⏳ Идет расчет вероятностей..."):
                probabilities = system.calculate()
        except Exception as e:
            st.error(f"Ошибка при расчёте: {e}")
            st.stop()

        st.markdown("---")
        st.subheader("📊 Графики вероятностей состояний системы")
        fig = plot_probabilities(probabilities, time_array)
        st.plotly_chart(fig, use_container_width=True)


main()
