"""Страница для моделирования СМО с нетерпеливыми заявками.

Позволяет пользователю задавать параметры системы, выбирать тип СМО, производить расчет
вероятностей и визуализировать результаты.
"""

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st

from app.domain import ComputationConfig, SystemMode, SystemType
from app.domain.models import (
    BaseSystemParams,
    CalculationSettings,
    MAPSystemParams,
    SystemParams,
    TransientSystemParams,
)
from app.services.systems import system_factory
from pages.components import (
    get_intensity_parameters,
    get_system_mode_type,
    map_intensity_matrix,
    plot_metric,
    plot_probabilities,
    render_calculation_config,
    render_initial_conditions,
    render_time_settings,
    system_capacity_inputs,
)

PLOT_SETTINGS = {
    "throughput": {
        "title_text": "Пропускная способность системы",
        "yaxis_title": "Заявки/ед. времени",
        "line_colors": ["#1f77b4"],
    },
    "avg_buffer_length": {
        "title_text": "Среднее число заявок в буфере",
        "yaxis_title": "Каналы",
        "line_colors": ["#ff7f0e"],
    },
    "avg_system_length": {
        "title_text": "Среднее число заявок в системе",
        "yaxis_title": "Заявки",
        "line_colors": ["#9467bd"],
    },
    "absolute_throughput": {
        "title_text": "Абсолютная пропускная способность",
        "yaxis_title": "Заявки/ед. времени",
        "line_colors": ["#2ca02c"],
    },
    "relative_throughput": {
        "title_text": "Относительная пропускная способность",
        "yaxis_title": "Доля от входящего потока",
        "line_colors": ["#d62728"],
    },
    "rejection_probability": {
        "title_text": "Вероятность отказа",
        "yaxis_title": "Вероятность",
        "line_colors": ["#8c564b"],
    },
    "loss_probability": {
        "title_text": "Вероятность потери пакетов в момент времени",
        "yaxis_title": "Вероятность",
        "line_colors": ["#17becf"],
    },
    "service_probability": {
        "title_text": "Вероятность обслуживания заявок",
        "yaxis_title": "Вероятность",
        "line_colors": ["#e377c2"],
    },
    "quit_probability": {
        "title_text": "Вероятность ухода заявки из системы",
        "yaxis_title": "Вероятность",
        "line_colors": ["#7f7f7f"],
    },
}


# pylint: disable=too-many-locals
def get_user_inputs() -> tuple[ComputationConfig, SystemParams]:
    """Собирает все входные параметры от пользователя через UI.

    Returns:
        tuple:
            - config (ComputationConfig): Конфигурация для выполнения вычислений.
            - params (ServerParams): Параметры выбранноСМО.
    """
    config = render_calculation_config()
    st.markdown("---")
    system_type, calculation_mode, system_mode = get_system_mode_type()

    st.subheader("⚙️ Параметры системы")
    lambda_rate, mu_rate, nu_rate = get_intensity_parameters(system_type)

    p_rate, q_rate = None, None
    if system_type in (SystemType.MAP, SystemType.MULTI_SENSOR_MAP):
        max_customers, processor_count = system_capacity_inputs(
            system_type, default_max_customers=3
        )

        if system_type == SystemType.MULTI_SENSOR_MAP:
            sensor_count = st.number_input(
                "Число датчиков для MAP-процесса (M)",
                min_value=1,
                max_value=100,
                value=3,
            )
            max_customers += 2

        st.markdown("---")

        p_rate, q_rate = map_intensity_matrix(
            max_customers if system_type == SystemType.MAP else sensor_count
        )
    else:
        max_customers, processor_count = system_capacity_inputs(system_type)

    count = max_customers

    if system_type == SystemType.MULTI and processor_count is not None:
        count += processor_count + 1
    elif system_type == SystemType.MAP:
        count **= 2
    elif system_type == SystemType.MULTI_SENSOR_MAP:
        count *= sensor_count

    transient_params = None
    if system_mode == SystemMode.TRANSIENT:
        st.markdown("---")
        state_variables, initial_probabilities = render_initial_conditions(count)

        st.markdown("---")
        time_array = render_time_settings()

        transient_params = TransientSystemParams(
            time_array=time_array,
            state_variables=state_variables,
            initial_probabilities=initial_probabilities,
        )

    base_params = BaseSystemParams(
        mu_rate=mu_rate,
        nu_rate=nu_rate,
        lambda_rate=lambda_rate,
        max_customers=max_customers,
        processor_count=processor_count,
    )

    settings = CalculationSettings(
        calculation_mode=calculation_mode,
        system_mode=system_mode,
        system_type=system_type,
    )

    if (
        system_type in (SystemType.MAP, SystemType.MULTI_SENSOR_MAP)
        and p_rate is not None
        and q_rate is not None
    ):
        base_params = MAPSystemParams(
            **asdict(base_params),
            p_rate=p_rate.astype(np.float64),
            q_rate=q_rate.astype(np.float64),
        )

    params = SystemParams(
        base_params=base_params,
        transient_params=transient_params,
        settings=settings,
    )
    return config, params


def format_dataframe(df: pd.DataFrame, precision: int = 16):
    """Форматирует все числовые значения в DataFrame.

    Args:
        df (pd.DataFrame): Исходный DataFrame.
        precision (int): Количество знаков после запятой для форматирования чисел.

    Returns:
        pd.io.formats.style.Styler: Отформатированный Styler для отображения
            в Streamlit.
    """
    return df.style.format(
        lambda x: f"{x:.{precision}f}" if isinstance(x, float) else x
    )


# ruff: noqa: C901
def main() -> None:
    def numpy_to_python(obj):
        """Рекурсивно преобразует numpy-типы и массивы в стандартные Python-типы."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [numpy_to_python(item) for item in obj]
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None  # Заменяем NaN/Inf на null для JSON
       
        return obj



    """Основная функция страницы: UI, вычисление, визуализация."""
    st.title("🧪 Моделирование СМО с нетерпеливыми заявками")
    st.markdown("---")

    try:
        config, params = get_user_inputs()
    except ValueError as e:
        st.error(f"❌ Ошибка в вводных данных: {e}")
        st.stop()

    if st.button("🚀 Применить параметры"):
        try:
            config.validate()
            params.validate()
        except ValueError as e:
            st.error(f"❌ Ошибка в параметрах: {e}")
            st.stop()

        # steady_system = system_factory(
        #     system_mode=SystemMode.STEADY,
        #     system_type=params.settings.system_type,
        #     params=params,
        #     config=config,
        # )
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
                # steady_system.calculate_probabilities()
                if transient_system is not None:
                    transient_system.calculate_probabilities()
        except ValueError as e:
            st.error(f"❌ Ошибка при вычислении: {e}")
            st.stop()

        st.markdown("---")

        if transient_system is not None:
            st.subheader("📈 Визуализация динамики состояний системы")
            results = transient_system.calculate()
            st.plotly_chart(
                plot_probabilities(
                    transient_system.probabilities,
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
            st.subheader("💾 Скачать результаты")
                
            try:
                results_serializable = numpy_to_python(results)
                json_data = json.dumps(
                    results_serializable,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False  # Запрещаем NaN/Inf в JSON
                )
                
                st.download_button(
                    label="📥 Скачать результаты в JSON",
                    data=json_data,
                    file_name="simulation_results.json",
                    mime="application/json",
                    key="download_json_button"
                )
            except Exception as e:
                st.warning(f"⚠️ Не удалось подготовить JSON для скачивания: {e}")
                st.info("Попробуйте экспортировать данные в другом формате или обратитесь к разработчику.")

        # st.subheader("📊 Вероятности стационарных состояний")
        # st.dataframe(format_dataframe(pd.DataFrame(probabilities)))

        # if params.settings.calculation_mode != CalculationMode.PROBABILITY:
        #     st.subheader("📈 Основные характеристики системы")
        #     st.dataframe(format_dataframe(pd.DataFrame(results)))


main()
