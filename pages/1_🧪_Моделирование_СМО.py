"""Моделирование СМО с нетерпеливыми заявками.

Этот модуль реализует интерактивное веб-приложение для численного моделирования
СМО типа M/M/m/n с нетерпеливыми заявками, которое может применяться для анализа
медицинских информационно-измерительных систем.

Основные функциональные возможности:
- Моделирование как одноканальных, так и многоканальных СМО
- Настройка параметров системы (интенсивности поступления, обслуживания, ухода)
- Визуализация динамики вероятностей состояний системы
- Проверка корректности входных параметров
"""

import numpy as np
import pandas as pd
import streamlit as st

from app.models import (
    CalculationType,
    ComputationConfig,
    MergedComputationConfig,
    MpmathComputationConfig,
    MultiServerParams,
    SingleServerParams,
)
from app.services.probability import MultiServerSystem, SingleServerSystem
from app.utils.plot import plot_probabilities

st.title("🧪 Моделирование СМО с нетерпеливыми заявками")

st.markdown(
    """
## 🔍 Описание
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

with st.expander("ℹ️ Как использовать это приложение"):
    st.write(
        """
    1. Задайте параметры системы в форме ниже
    2. Укажите временной диапазон для моделирования
    3. Задайте начальные вероятности состояний
    4. Нажмите "Применить параметры"
    5. Используйте график вероятностей, который будет отображен ниже
    """
    )

st.markdown("---")
with st.form("settiongs_form"):
    st.subheader("⚙️ Настройки вычисления для системы")
    calculation_type = st.selectbox(
        "Выберите тип вычисления:",
        [
            "Numpy (быстрое, для небольших систем)",
            "Mpmath (точное, для больших систем)",
            "Объединенное (для сложных систем с высокой точностью)",
        ],
    )
    if calculation_type == "Numpy (быстрое, для небольших систем)":
        config = ComputationConfig(calculation_type=CalculationType.NUMPY)
    elif calculation_type == "Mpmath (точное, для больших систем)":
        config = MpmathComputationConfig(calculation_type=CalculationType.MPMATH)
    else:
        config = MergedComputationConfig(calculation_type=CalculationType.MERGED)
    submitted = st.form_submit_button("✅ Применить настройки")

if submitted:
    st.success("✅ Настройки успешно применены!")
    st.session_state.config = config

st.markdown("---")
st.subheader("🔬 Тип системы")

system_type = st.selectbox("Выберите тип СМО:", ["Однолинейная", "Многолинейная"])

with st.form("param_form"):
    st.subheader("⚙️ Параметры системы")

    col1, col2, col3 = st.columns(3)
    with col1:
        lambda_rate = st.number_input(
            "Интенсивность поступления заявок (λ)",
            min_value=0.0,
            value=8333.0,
            format="%.10f",
        )
    with col2:
        mu_rate = st.number_input(
            "Интенсивность обслуживания заявок (μ)",
            min_value=0.0,
            value=10833.0,
            format="%.10f",
        )
    with col3:
        nu_rate = st.number_input(
            "Интенсивность ухода нетерпеливых заявок (ν)",
            min_value=0.0,
            value=12345.0,
            format="%.10f",
        )

    st.markdown("---")

    max_customers = st.number_input(
        "Максимальное количество заявок в системе (n)",
        min_value=1,
        max_value=100,
        value=4,
    )

    if system_type == "Многолинейная":
        processor_count = st.number_input(
            "Количество обслуживающих процессоров (m)",
            min_value=1,
            max_value=100,
            value=2,
        )

    st.subheader("⏳ Временные параметры")
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        t_start = st.number_input(
            "Начальное время", min_value=0.0, value=0.0, format="%.10f"
        )
    with col_t2:
        t_end = st.number_input(
            "Конечное время", min_value=0.0, value=0.001, format="%.10f"
        )
    with col_t3:
        t_steps = st.number_input(
            "Количество шагов по времени", min_value=10, value=1000
        )
    time_array = np.linspace(t_start, t_end, int(t_steps), dtype=np.float64)

    st.subheader("📈 Начальные условия")
    state_variables_str = st.text_input(
        "Переменные состояния — *введите через запятую*", "1, 1, 1, 1"
    )
    state_variables = np.array(
        [float(x.strip()) for x in state_variables_str.split(",")]
    )

    initial_probabilities_str = st.text_input(
        "Начальные вероятности — *введите через запятую*", "1, 0, 0, 0"
    )
    initial_probabilities = np.array(
        [float(x.strip()) for x in initial_probabilities_str.split(",")]
    )

    submitted = st.form_submit_button("🚀 Применить параметры")

if submitted:
    # Валидация входных данных
    if not np.isclose(np.sum(initial_probabilities), 1.0):
        st.error("Сумма начальных вероятностей должна быть равна 1.")
        st.stop()

    if len(state_variables) != len(initial_probabilities):
        st.error(
            "Количество переменных состояния должно совпадать с количеством начальных\
                вероятностей."
        )
        st.stop()

    if np.any(initial_probabilities < 0) or np.any(initial_probabilities > 1):
        st.error("Все вероятности должны быть в диапазоне от 0 до 1.")
        st.stop()

    if np.any(state_variables < 0):
        st.error("Переменные состояния не могут быть отрицательными.")
        st.stop()

    if t_end <= t_start:
        st.error("Конечное время должно быть больше начального.")
        st.stop()

    if lambda_rate <= 0 or mu_rate <= 0 or nu_rate <= 0:
        st.error("Все интенсивности (λ, μ, ν) должны быть положительными.")
        st.stop()

    # Инициализация соответствующей системы
    impatient_queue_system: SingleServerSystem | MultiServerSystem
    if system_type == "Многолинейная":
        params_for_multi = MultiServerParams(
            lambda_rate=lambda_rate,
            mu_rate=mu_rate,
            nu_rate=nu_rate,
            max_customers=max_customers,
            processor_count=processor_count,
            time_array=time_array,
            state_variables=state_variables,
            initial_probabilities=initial_probabilities,
        )
        impatient_queue_system = MultiServerSystem(params_for_multi, config)
    else:
        params = SingleServerParams(
            lambda_rate=lambda_rate,
            mu_rate=mu_rate,
            nu_rate=nu_rate,
            max_customers=max_customers,
            time_array=time_array,
            state_variables=state_variables,
            initial_probabilities=initial_probabilities,
        )
        impatient_queue_system = SingleServerSystem(params, config)

    st.success("✅ Параметры успешно заданы!")

    # Расчет и визуализация результатов
    with st.spinner("⏳ Идет расчет вероятностей..."):
        probabilities = impatient_queue_system.calculate()

        st.subheader("📊 Графики вероятностей состояний системы")
        fig = plot_probabilities(probabilities, time_array)
        st.pyplot(fig)

        st.subheader("📝 Вероятности в конце периода")
        end_probs = probabilities[:, -1]
        prob_df = pd.DataFrame(
            {
                "Состояние": [f"S{i}" for i in range(len(end_probs))],
                "Вероятность": end_probs,
            }
        )
        st.table(prob_df)

        st.markdown(f"**📌 Сумма итоговых вероятностей:** {sum(end_probs):.6f}")
