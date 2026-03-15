"""Содержит функции для подготовки переходной системы и отображения результатов моделирования.

Модуль отвечает за:
    - валидацию конфигурации вычислений и параметров системы;
    - создание экземпляра системы для переходного режима;
    - запуск предварительного расчёта вероятностей состояний;
    - визуализацию вероятностей и вычисленных метрик;
    - экспорт результатов моделирования в JSON.

Основные функции:
    prepare_transient_system() — проверяет параметры, создаёт систему и
    выполняет предварительный расчёт для переходного режима.

    render_results() — отображает графики результатов моделирования и
    добавляет кнопку скачивания данных в формате JSON.
"""

import streamlit as st

from app.domain import ComputationConfig, SystemMode
from app.domain.models import (
    SystemParams,
)
from app.services.systems import system_factory
from pages.components import (
    plot_metric,
    plot_probabilities,
)

from .constants import PLOT_SETTINGS
from .serializers import results_to_json


def prepare_transient_system(
    config: ComputationConfig,
    params: SystemParams,
):
    """Проверяет параметры, создаёт систему и запускает предварительный расчёт.

    Args:
        config (ComputationConfig): Конфигурация вычислений.
        params (SystemParams): Параметры системы.

    Returns:
        Any | None: Экземпляр переходной системы, если выбран режим TRANSIENT.
        В противном случае — None.
    """
    try:
        config.validate()
        params.validate()
    except ValueError as e:
        st.error(f"❌ Ошибка в параметрах: {e}")
        st.stop()

    transient_system = None
    if params.settings.system_mode == SystemMode.TRANSIENT:
        transient_system = system_factory(
            system_mode=SystemMode.TRANSIENT,
            system_type=params.settings.system_type,
            params=params,
            config=config,
        )

    try:
        with st.spinner("⏳ Идёт расчёт значений..."):
            if transient_system is not None:
                transient_system.calculate_probabilities()
    except ValueError as e:
        st.error(f"❌ Ошибка при вычислении: {e}")
        st.stop()

    return transient_system


def render_results(system, params: SystemParams) -> None:
    """Отображает результаты режима и кнопку экспорта JSON.

    Args:
        system: Объект системы.
        params (SystemParams): Параметры системы.

    Returns:
        None: Функция ничего не возвращает.
    """
    if system is None:
        return

    st.subheader("📈 Визуализация динамики состояний системы")
    results = system.calculate()

    st.plotly_chart(
        plot_probabilities(
            system.probabilities,
            params.transient_params.time_array,
            params.settings.calculation_mode.value,
        ),
        width="stretch",
    )

    for key, value in results.items():
        try:
            if key == "probability":
                continue

            st.plotly_chart(
                plot_metric(
                    value,
                    params.transient_params.time_array,
                    **PLOT_SETTINGS[key],
                ),
                width="stretch",
            )
        except Exception as e:
            st.error(f"❌ Ошибка в результатах: {e}")

    st.markdown("---")

    try:
        json_data = results_to_json(results, params.transient_params.time_array)

        st.download_button(
            label="📥 Скачать результаты в JSON",
            data=json_data,
            file_name="simulation_results.json",
            mime="application/json",
            key="download_json_button",
        )
    except Exception as e:
        st.warning(f"⚠️ Не удалось подготовить JSON для скачивания: {e}")
        st.info(
            "Попробуйте экспортировать данные в другом формате или обратитесь к разработчику."
        )
