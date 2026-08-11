"""Страница трёхмерной визуализации характеристик СМО.

Позволяет задать параметры системы, рассчитать её характеристики и для
каждой характеристики выбрать вид третьей оси: развёртку по параметру
системы, вторую метрику или номер датчика.
"""

import streamlit as st
from pages.plots3d import render_page
from progress import StreamlitProgressStrategy

from quick.utils.progress import Progress


def main() -> None:
    """Основная функция страницы: UI, вычисление, визуализация."""
    Progress.set_strategy(StreamlitProgressStrategy())

    st.title("🧊 Трёхмерные графики характеристик СМО")
    st.markdown("---")

    render_page()


if __name__ == "__main__":
    main()
