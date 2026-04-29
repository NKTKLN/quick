"""Модуль Streamlit-компонентов для задания параметров интенсивности и емкости системы.

Содержит функции для ввода пользователем:
- Интенсивностей потоков заявок, обслуживания и ухода;
- Максимальной емкости системы и количества процессоров.
"""

from ast import literal_eval
from typing import cast

import numpy as np
import streamlit as st
from numpy.typing import NDArray

from quick.domain import SystemMode, SystemType
from quick.domain.models import TimeSeries, TimeSeriesBaseSystemParams
from quick.utils import map_intensity_matrix_generator


def get_system_mode_type() -> tuple[SystemType, SystemMode]:
    """Отображает UI-компонент с двумя полями выбора (тип системы и режим вычисления).

    Returns:
        tuple[SystemType, SystemMode]: Значения типа системы и режима системы.
    """

    def reset_state() -> None:
        """Очищает состояние Streamlit."""
        keys_to_delete = [k for k in st.session_state if k != "system_type"]
        for k in keys_to_delete:
            del st.session_state[k]

    st.subheader("🔬 Тип системы")
    system_type = st.selectbox(
        "Выберите тип СМО:",
        [system_type.value for system_type in SystemType],
        key="system_type",
        on_change=reset_state,
    )
    system_mode = st.selectbox(
        "Выберите режим системы:", [mode.value for mode in SystemMode]
    )
    return (
        SystemType(system_type),
        SystemMode(system_mode),
    )


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

    Returns:
        tuple: Параметры интенсивности, соответствующие выбранному типу системы
            и режиму расчета.
    """
    if system_type == SystemType.MAP:
        return map_intensity_parameters()
    return intensity_parameters()


def _generate_matrix(
    n: int,
    title: str,
    key: str,
    default_matrix: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Отображает UI-компонент с матрицей для редактирования пользователем.

    Args:
        n (int): Размерность квадратной матрицы (n x n).
        title (str): Заголовок для компонента разворачиваемого блока.
        key (str): Уникальный ключ для Streamlit компонента редактирования таблицы.
        default_matrix (NDArray[np.float64] | None): Начальная матрица значений.
            Если None, используется матрица из нулей размером n x n.

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


def _parse_array(text: str, field_name: str) -> np.ndarray:
    """Преобразует строковое представление массива в numpy.ndarray.

    Args:
        text (str): Строка с представлением массива (например, "[1, 2, 3]"
            или "[[1, 2], [3, 4]]").
        field_name (str): Имя поля (используется в сообщениях об ошибках).

    Returns:
        np.ndarray: Массив numpy с типом float64.

    Raises:
        ValueError: Если строка не может быть преобразована в массив,
            если массив имеет размерность, отличную от 1D или 2D,
            или если массив пустой.
    """
    try:
        value = literal_eval(text)
        arr = np.array(value, dtype=np.float64)

        if arr.ndim not in (1, 2):
            raise ValueError(f"{field_name} должен быть 1D или 2D массивом")

        if arr.size == 0:
            raise ValueError(f"{field_name} не должен быть пустым")

        return arr
    except Exception as e:
        raise ValueError(f"Ошибка в поле '{field_name}': {e}") from e


def get_time_series_parameters() -> TimeSeriesBaseSystemParams:
    """Отображает UI-компоненты для ввода параметров, зависящих от времени.

    Позволяет пользователю задать временные ряды для интенсивностей:
    - поступления заявок (λ),
    - обслуживания (μ),
    - ухода нетерпеливых заявок (ν).

    Returns:
        TimeSeriesBaseSystemParams: Объект с параметрами временных рядов.
            Если пользователь отключил использование временных рядов,
            возвращается объект с параметами по умолчанию.

    Raises:
        StreamlitAPIException: Прерывает выполнение (st.stop),
            если введённые данные некорректны и не могут быть
            преобразованы в массивы.
    """
    st.subheader("🕔 Параметры, зависимые от времени")

    time_series_params = TimeSeriesBaseSystemParams()

    use_timeseries = st.checkbox(
        "Использовать параметры, зависимые от времени", value=True
    )
    if not use_timeseries:
        return time_series_params

    lambda_times_text = st.text_area(
        "lambda_times",
        value="[0.045000, 0.092000, 0.138000, 0.184000, 0.231000, 0.277000, 0.323000, 0.369000, 0.416000, 0.462000]",
    )

    lambda_values_text = st.text_area(
        "lambda_values",
        value="""[
    [169.675095, 447.755615, 936.126038],
    [210.134125, 454.556976, 952.228271],
    [209.509659, 454.173676, 951.484375],
    [206.140350, 450.753174, 952.228271],
    [209.509659, 454.173676, 951.484375],
    [207.964615, 454.556976, 951.484375],
    [204.347824, 452.647064, 947.045105],
    [210.134125, 454.556976, 951.484375],
    [209.198807, 453.791077, 950.741577],
    [210.447769, 452.647064, 952.228271]
]""",
    )

    mu_times_text = st.text_area(
        "mu_times",
        value="[0.045000, 0.092000, 0.138000, 0.184000, 0.231000, 0.277000, 0.323000, 0.369000, 0.416000, 0.462000]",
    )

    mu_values_text = st.text_area(
        "mu_values",
        value="[509.308807, 510.693085, 509.832611, 509.854919, 509.877228, 509.709900, 510.078186, 510.044678, 509.866058, 510.066986]",
    )

    nu_times_text = st.text_area(
        "nu_times",
        value="[0.045000, 0.092000, 0.138000, 0.184000, 0.231000, 0.277000, 0.323000, 0.369000, 0.416000, 0.462000]",
    )

    nu_values_text = st.text_area(
        "nu_values",
        value="[100.000000, 100.000000, 100.000000, 100.000000, 100.000000, 100.000000, 100.000000, 100.000000, 100.000000, 100.000000]",
    )

    try:
        lambda_times = _parse_array(lambda_times_text, "lambda_times")
        lambda_values = _parse_array(lambda_values_text, "lambda_values")
        mu_times = _parse_array(mu_times_text, "mu_times")
        mu_values = _parse_array(mu_values_text, "mu_values")
        nu_times = _parse_array(nu_times_text, "nu_times")
        nu_values = _parse_array(nu_values_text, "nu_values")

        time_series_params.lambda_rate = TimeSeries(lambda_times, lambda_values)
        time_series_params.mu_rate = TimeSeries(mu_times, mu_values)
        time_series_params.nu_rate = TimeSeries(nu_times, nu_values)

    except ValueError as e:
        st.error(str(e))
        st.stop()

    return time_series_params
