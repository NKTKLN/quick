"""Страница для моделирования СМО с нетерпеливыми заявками.

Позволяет пользователю задавать параметры системы, выбирать тип СМО, производить расчет
вероятностей и визуализировать результаты.
"""

import streamlit as st
from pages.simulation import (
    get_user_inputs,
    prepare_system,
    render_steady_table,
    render_transient_results,
)

from quick.domain.enums import SystemMode


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
        system = prepare_system(config, params)

        st.markdown("---")
        if params.calculation_params.system_mode == SystemMode.TRANSIENT:
            render_transient_results(system, params)
        elif params.calculation_params.system_mode == SystemMode.STEADY:
            render_steady_table(system)


if __name__ == "__main__":
    main()
