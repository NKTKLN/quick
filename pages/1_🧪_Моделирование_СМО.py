"""Страница для моделирования СМО с нетерпеливыми заявками.

Позволяет пользователю задавать параметры системы, выбирать тип СМО, производить расчет
вероятностей и визуализировать результаты.
"""

import streamlit as st

from pages.simulation import get_user_inputs, prepare_transient_system, render_results


def main() -> None:
    """Основная функция страницы: UI, вычисление, визуализация."""
    st.title("🧪 Моделирование СМО с нетерпеливыми заявками")
    st.markdown("---")

    try:
        config, params = get_user_inputs()
    except ValueError as e:
        st.error(f"❌ Ошибка в вводных данных: {e}")
        st.stop()

    if st.button("🚀 Применить параметры"):
        transient_system = prepare_transient_system(config, params)

        st.markdown("---")
        render_results(transient_system, params)


if __name__ == "__main__":
    main()
