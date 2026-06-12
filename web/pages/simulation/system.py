"""Содержит функции для подготовки системы и отображения результатов моделирования.

Модуль отвечает за:
    - валидацию конфигурации вычислений и параметров системы;
    - создание экземпляра системы для переходного или стационарного режима;
    - запуск предварительного расчёта вероятностей состояний;
    - визуализацию вероятностей и вычисленных метрик для переходного режима;
    - отображение таблицы стационарных состояний для стационарного режима;
    - экспорт результатов моделирования в JSON.

Основные функции:
    prepare_system() — проверяет параметры, создаёт систему и
    выполняет предварительный расчёт вероятностей состояний.

    render_transient_results() — отображает графики результатов моделирования
    для переходного режима и добавляет кнопку скачивания данных в формате JSON.

    render_steady_table() — отображает таблицу стационарных состояний
    системы с индексами состояний и значениями вероятностей без округления.
"""

import numpy as np
import pandas as pd
import streamlit as st
from pages.components import (
    plot_metric,
    plot_probabilities,
)

from quick.domain import ComputationConfig, SystemMode
from quick.domain.params import SystemParams
from quick.services.systems import system_factory
from quick.services.systems.base import BaseServerSystem
from quick.services.systems.steady_state import MultiServerSteadyStateSystem

from .constants import PLOT_SETTINGS
from .serializers import results_to_json


def prepare_system(
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

    system = None
    if params.calculation_params.system_mode == SystemMode.TRANSIENT:
        system = system_factory(
            system_mode=SystemMode.TRANSIENT,
            system_type=params.calculation_params.system_type,
            params=params,
            config=config,
        )
    elif params.calculation_params.system_mode == SystemMode.STEADY:
        system = system_factory(
            system_mode=SystemMode.STEADY,
            system_type=params.calculation_params.system_type,
            params=params,
            config=config,
        )

    try:
        with st.spinner("⏳ Идёт расчёт значений..."):
            if system is not None:
                system.calculate_probabilities()
    except ValueError as e:
        st.error(f"❌ Ошибка при вычислении: {e}")
        st.stop()

    return system


def render_transient_results(system: BaseServerSystem, params: SystemParams) -> None:
    """Отображает результаты режима и кнопку экспорта JSON.

    Args:
        system (BaseServerSystem): Объект системы.
        params (SystemParams): Параметры системы.

    Returns:
        None: Функция ничего не возвращает.
    """
    st.subheader("📈 Визуализация динамики состояний системы")
    results = system.calculate()

    st.plotly_chart(
        plot_probabilities(
            system.probabilities,
            params.transient_params.time_array,
            "Вероятности",
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


def render_steady_table(system: MultiServerSteadyStateSystem) -> None:
    """Отображает таблицу стационарных состояний системы.

    В таблице:
        - первый столбец: индекс состояния от 0 до n-1;
        - второй столбец: значение вероятности без округления.

    Args:
        system (BaseServerSystem): Объект системы.

    Returns:
        None
    """
    probabilities = getattr(system, "probabilities", None)
    if probabilities is None:
        st.warning("⚠️ Вероятности стационарных состояний отсутствуют.")
        return

    try:
        probabilities_array = np.asarray(probabilities).reshape(-1)

        df = pd.DataFrame(
            {
                "Индекс": np.arange(len(probabilities_array)),
                "Значение": probabilities_array,
            }
        )

        df["Значение"] = df["Значение"].map(lambda x: f"{x:.15e}")

        st.subheader("Стационарные состояния системы")
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception as e:
        st.error(f"❌ Ошибка при отображении стационарных состояний: {e}")
