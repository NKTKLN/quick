"""Компоненты Streamlit для ввода переменных состояния и начальных вероятностей.

Позволяет пользователю ввести значения через текстовое поле, с валидацией
и автоматическим заполнением значениями по умолчанию.
"""

import numpy as np
import streamlit as st


def _all_values_equal(
    values_str: str, target_value: int = 1, min_count: int = 1, skip_first: bool = False
) -> bool:
    """Проверяет, равны ли все значения в строке заданному числу.

    Args:
        values_str (str): Строка значений, разделённых запятой.
        target_value (int): Число, которому должны быть равны все элементы.
        min_count (int): Минимальное количество элементов для проверки.
        skip_first (bool): Пропуск первого значения.

    Returns:
        bool: True, если все значения равны target_value и их не меньше min_count.
    """
    try:
        numbers = [int(x.strip()) for x in values_str.split(",")]
        if skip_first:
            numbers = numbers[1:]
        return len(numbers) >= min_count and all(n == target_value for n in numbers)
    except ValueError:
        return False


def render_state_variables(count: int) -> np.ndarray[np.float64]:
    """Отображает UI-компонент для ввода переменных состояния.

    Args:
        count (int): Ожидаемое количество переменных.

    Returns:
        np.ndarray[np.float64]: Массив введённых переменных состояния.
    """
    if "state_variables" not in st.session_state or _all_values_equal(
        st.session_state.state_variables
    ):
        st.session_state.state_variables = ", ".join(["1"] * count)

    st.session_state.state_variables = st.text_input(
        "Переменные состояния — *введите через запятую* (например: 1, 2, 3)",
        st.session_state.state_variables,
        key="state_variables_input",
    )

    try:
        state_variables = np.array(
            [float(x.strip()) for x in st.session_state.state_variables.split(",")]
        )
    except ValueError:
        st.error("Ошибка: введите корректные числа, разделённые запятыми.")
        return np.zeros(count)

    return state_variables


def render_initial_probabilities(count: int) -> np.ndarray[np.float64]:
    """Отображает UI-компонент для ввода начальных вероятностей.

    Args:
        count (int): Ожидаемое количество вероятностей.

    Returns:
        np.ndarray[np.float64]: Массив начальных вероятностей.
    """
    if "initial_probabilities" not in st.session_state or _all_values_equal(
        st.session_state.initial_probabilities, target_value=0, skip_first=True
    ):
        st.session_state.initial_probabilities = "1" + ", 0" * (count - 1)

    st.session_state.initial_probabilities = st.text_input(
        "Начальные вероятности — *введите через запятую* (например: 1, 0, 0, 0)",
        st.session_state.initial_probabilities,
        key="initial_probabilities_input",
    )

    try:
        initial_probabilities = np.array(
            [
                float(x.strip())
                for x in st.session_state.initial_probabilities.split(",")
            ]
        )
    except ValueError:
        st.error("Ошибка: введите корректные числовые значения через запятую.")
        return np.zeros(count)

    return initial_probabilities
