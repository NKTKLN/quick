"""Компонент Streamlit для задания временных параметров моделирования.

Позволяет пользователю выбрать начальное и конечное время,
а также количество шагов, после чего формируется массив временных отсечек.
"""

import numpy as np
import streamlit as st


def time_settings() -> np.ndarray[np.float64]:
    """Отображает UI-компонент для ввода временных параметров.

    Пользователь задаёт:
      - начальное время моделирования (t_start),
      - конечное время моделирования (t_end),
      - количество временных шагов (t_steps).

    Returns:
        np.ndarray: Массив временных равномерно распределённых отсечек.
    """
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
    return time_array
