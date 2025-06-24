"""Модуль Streamlit-компонентов для задания параметров интенсивности и емкости системы.

Содержит функции для ввода пользователем:
- Интенсивностей потоков заявок, обслуживания и ухода;
- Максимальной емкости системы и количества процессоров.
"""

from typing import Optional, cast

import numpy as np
import streamlit as st
from numpy.typing import NDArray

from app.domain import CalculationMode, SystemType
from app.services import map_intensity_matrix_generator


def get_system_mode_type() -> tuple[SystemType, CalculationMode]:
    """Отображает UI-компонент с двумя полями выбора (тип системы и режим вычисления).

    Returns:
        tuple[SystemType, CalculationMode]: Значения типа системы и режима вычисления.
    """
    st.subheader("🔬 Тип системы")
    system_type = st.selectbox(
        "Выберите тип СМО:", [system_type.value for system_type in SystemType]
    )
    calculation_mode = st.selectbox(
        "Выберите режим расчёта:", [mode.value for mode in CalculationMode]
    )
    return SystemType(system_type), CalculationMode(calculation_mode)


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


def map_intensity_parameters() -> tuple[NDArray[np.float64], float, float]:
    """Отображает UI-компонент с параметрами интенсивности: ν, μ и массивом λ.

    Returns:
        tuple[NDArray[np.float64], float, float]: Значения интенсивности поступления
            заявок (λ) в виде массива, интенсивности обслуживания (μ) и интенсивности
            ухода нетерпеливых заявок (ν).
    """
    col1, col2 = st.columns(2)

    with col1:
        mu_rate = st.number_input(
            "Интенсивность обслуживания заявок (μ)",
            min_value=0.0,
            value=500.0,
            format="%.10f",
        )
    with col2:
        nu_rate = st.number_input(
            "Интенсивность ухода нетерпеливых заявок (ν)",
            min_value=0.0,
            value=100.0,
            format="%.10f",
        )

    lambda_rate_str = st.text_input(
        "Интенсивность поступления заявок (λ) — *введите через запятую*",
        value="850, 8000, 67400",
        placeholder="Например: 850, 8000, 67400",
    )

    # Преобразуем введённую строку в массив float, игнорируя пустые элементы
    lambda_rate = np.array(
        [float(x.strip()) for x in lambda_rate_str.split(",") if x.strip()]
    )

    return lambda_rate, mu_rate, nu_rate


def get_intensity_parameters(system_type: SystemType) -> tuple:
    """Возвращает параметры интенсивности в зависимости от параметров системы.

    Args:
        system_type (SystemType): Тип системы.
        calculation_mode (CalculationMode): Режим расчета.

    Returns:
        tuple: Параметры интенсивности, соответствующие выбранному типу системы
            и режиму расчета.
    """
    if system_type == SystemType.MAP:
        return map_intensity_parameters()
    return intensity_parameters()


def system_capacity_inputs(
    system_type: SystemType,
    default_max_customers: int = 4,
    default_processor_count: int = 2,
) -> tuple[int, Optional[int]]:
    """Отображает UI-компонент для ввода емкости системы и количества процессоров.

    Args:
        system_type (SystemType): Тип системы. Если "Многолинейная", появляется поле
            для количества процессоров.
        default_max_customers (int, optional): Значение по умолчанию для максимального
            количества заявок в системе. По умолчанию 4.
        default_processor_count (int, optional): Значение по умолчанию для количества
            процессоров в многолинейной системе. По умолчанию 2.

    Returns:
        tuple[int, Optional[int]]: Максимальное количество заявок (n) и количество
            процессоров (m), если применимо.
    """
    max_customers = st.number_input(
        "Максимальное количество заявок в системе (n)",
        min_value=1,
        max_value=100,
        value=default_max_customers,
    )

    processor_count = None
    if system_type == SystemType.MULTI:
        processor_count = st.number_input(
            "Количество обслуживающих процессоров (m)",
            min_value=1,
            max_value=100,
            value=default_processor_count,
        )
    return max_customers, processor_count


def _generate_matrix(
    n: int,
    title: str,
    key: str,
    default_matrix: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """Отображает UI-компонент с матрицей для редактирования пользователем.

    Args:
        n (int): Размерность квадратной матрицы (n x n).
        title (str): Заголовок для компонента разворачиваемого блока.
        key (str): Уникальный ключ для Streamlit компонента редактирования таблицы.
        default_matrix (Optional[NDArray[np.float64]], optional): Начальная матрица
            значений. Если None, используется матрица из нулей размером n x n.
            По умолчанию None.

    Returns:
        NDArray[np.float64]: Матрица интенсивностей размером n x n.
    """
    if default_matrix is None:
        default_matrix = np.zeros((n, n), dtype=np.float64)

    if key not in st.session_state:
        st.session_state[key] = default_matrix

    if len(st.session_state[key]) != n:
        old_matrix = st.session_state[key]
        new_matrix = np.zeros((n, n), dtype=np.float64)
        min_n = min(len(st.session_state[key]), n)
        new_matrix[:min_n, :min_n] = old_matrix[:min_n, :min_n]
        st.session_state[key] = new_matrix

    with st.expander(title):
        matrix = st.data_editor(
            st.session_state[key], num_rows="fixed", hide_index=True, key=f"{key}_table"
        )

    st.session_state[key] = matrix

    return cast(NDArray[np.float64], matrix)


def map_intensity_matrix(
    max_customers: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Отображает UI-компоненты для ввода двух матриц интенсивностей.

    Args:
        max_customers (int): Максимальное количество заявок в
            системе (размерность матриц).

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Две матрицы
            интенсивностей размера max_customers x max_customers:
            - Матрица интенсивностей обслуживания (p_rate);
            - Матрица интенсивностей поступления (q_rate).
    """
    if st.button("🔄 Сгенерировать матрицы интенсивностей случайно"):
        try:
            numpy_p_rate, numpy_q_rate = map_intensity_matrix_generator(max_customers)
            st.session_state["p_rate"] = numpy_p_rate
            st.session_state["q_rate"] = numpy_q_rate
        except ValueError:
            st.error("❌ Ошибка при генерации матриц")

    p_rate = _generate_matrix(
        max_customers, "Матрица интенсивностей обслуживания", "p_rate"
    )
    q_rate = _generate_matrix(
        max_customers, "Матрица интенсивностей поступления", "q_rate"
    )

    return p_rate, q_rate
