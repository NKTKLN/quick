"""Модуль Streamlit-компонентов для задания параметров интенсивности и емкости системы.

Содержит функции для ввода пользователем:
- Интенсивностей потоков заявок, обслуживания и ухода;
- Максимальной емкости системы и количества процессоров.
"""

from typing import Optional

import numpy as np
import streamlit as st


def intensity_parameters() -> tuple[float, float, float]:
    """Отображает UI-компонент с тремя полями ввода параметров интенсивности: λ, μ и ν.

    Returns:
        tuple[float, float, float]: Значения интенсивности поступления заявок (λ),
            интенсивности обслуживания (μ) и
            интенсивности ухода нетерпеливых заявок (ν).
    """
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

    return lambda_rate, mu_rate, nu_rate


def throughput_intensity_parameters() -> tuple[float, float, np.ndarray[np.float64]]:
    """Отображает UI-компонент с параметрами интенсивности: λ, μ и массивом ν.

    Returns:
        tuple[float, float, np.ndarray[np.float64]]: Значения интенсивности поступления
            заявок (λ), интенсивности обслуживания (μ) и интенсивности
            ухода нетерпеливых заявок (ν) в виде массива.
    """
    col1, col2 = st.columns(2)

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

    nu_rate_str = st.text_input(
        "Интенсивность ухода нетерпеливых заявок (ν) — *введите через запятую*",
        value="1000, 10833, 1e6",
        placeholder="Например: 1000, 10833, 100000",
    )

    # Преобразуем введённую строку в массив float, игнорируя пустые элементы
    nu_rate = np.array([float(x.strip()) for x in nu_rate_str.split(",") if x.strip()])

    return lambda_rate, mu_rate, nu_rate


def system_capacity_inputs(system_type: str) -> tuple[int, Optional[int]]:
    """Отображает UI-компонент для ввода емкости системы и количества процессоров.

    Args:
        system_type (str): Тип системы. Если "Многолинейная", появляется поле
            для количества процессоров.

    Returns:
        tuple[int, Optional[int]]: Максимальное количество заявок (n) и количество
            процессоров (m), если применимо.
    """
    max_customers = st.number_input(
        "Максимальное количество заявок в системе (n)",
        min_value=1,
        max_value=100,
        value=4,
    )

    processor_count = None
    if system_type == "Многолинейная":
        processor_count = st.number_input(
            "Количество обслуживающих процессоров (m)",
            min_value=1,
            max_value=100,
            value=2,
        )
    return max_customers, processor_count
