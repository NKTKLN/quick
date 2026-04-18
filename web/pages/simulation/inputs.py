"""Содержит функции для сбора пользовательских параметров модели СМО через Streamlit UI.

Модуль отвечает за:
    - получение конфигурации вычислений;
    - выбор типа системы, режима расчёта и режима работы;
    - ввод интенсивностей и параметров ёмкости системы;
    - настройку MAP-процесса для соответствующих типов СМО;
    - подготовку параметров переходного режима;
    - формирование итогового объекта параметров SystemParams.

Основная функция:
    get_user_inputs() — собирает все параметры из пользовательского интерфейса
    и возвращает конфигурацию вычислений вместе с параметрами системы.
"""

import ast
from dataclasses import asdict

import numpy as np
import streamlit as st
from pages.components import (
    get_intensity_parameters,
    get_system_mode_type,
    map_intensity_matrix,
    render_calculation_config,
    render_initial_conditions,
    render_time_settings,
)

from quick.domain import ComputationConfig, SystemMode, SystemType
from quick.domain.enums import CalculationMethod
from quick.domain.models import (
    CalculationParams,
    SystemParams,
    TransientSystemParams,
)
from quick.domain.models.imitation_params import ImitationSystemParams, SimulationParams
from quick.domain.models.system_params import (
    BaseSystemParams,
    MAPSystemParams,
    MultiSystemParams,
)
from quick.domain.models.timeseries import TimeSeries, TimeSeriesBaseSystemParams


def get_user_inputs() -> tuple[ComputationConfig, SystemParams]:
    """Собирает все входные параметры от пользователя через UI.

    Returns:
        tuple[ComputationConfig, SystemParams]:
            - config (ComputationConfig): Конфигурация выполнения вычислений.
            - params (SystemParams): Параметры выбранной СМО.
    """
    config = render_calculation_config()
    st.markdown("---")

    system_type, system_mode = get_system_mode_type()
    st.subheader("⚙️ Параметры системы")

    lambda_rate, mu_rate, nu_rate = get_intensity_parameters(system_type)

    max_customers, processor_count, p_rate, q_rate, sensor_count = (
        _get_capacity_and_map_params(system_type)
    )

    state_count = _calculate_state_count(
        system_type=system_type,
        max_customers=max_customers,
        processor_count=processor_count,
        sensor_count=sensor_count,
    )

    transient_params = _get_transient_params(
        system_mode=system_mode,
        state_count=state_count,
    )

    base_params = _build_base_params(
        system_type=system_type,
        lambda_rate=lambda_rate,
        mu_rate=mu_rate,
        nu_rate=nu_rate,
        max_customers=max_customers,
        processor_count=processor_count,
        sensor_count=sensor_count,
        p_rate=p_rate,
        q_rate=q_rate,
    )

    settings = CalculationParams(
        system_mode=system_mode,
        system_type=system_type,
    )

    params = SystemParams(
        base_params=base_params,
        transient_params=transient_params,
        calculation_params=settings,
    )

    def parse_1d_array(text: str, field_name: str) -> np.ndarray:
        try:
            value = ast.literal_eval(text)
            arr = np.array(value, dtype=np.float64)

            if arr.ndim != 1:
                raise ValueError(f"{field_name} должен быть одномерным массивом")

            if arr.size == 0:
                raise ValueError(f"{field_name} не должен быть пустым")

            return arr
        except Exception as e:
            raise ValueError(f"Ошибка в поле '{field_name}': {e}")

    def parse_lambda_array(text: str, field_name: str) -> np.ndarray:
        try:
            value = ast.literal_eval(text)
            arr = np.array(value, dtype=np.float64)

            if arr.ndim not in (1, 2):
                raise ValueError(f"{field_name} должен быть 1D или 2D массивом")

            if arr.size == 0:
                raise ValueError(f"{field_name} не должен быть пустым")

            return arr
        except Exception as e:
            raise ValueError(f"Ошибка в поле '{field_name}': {e}")

    if config.calculation_method == CalculationMethod.IMITATION:
        st.subheader("⚠️ Имитационные параметры (ВРЕМЕННО)")

        col1, col2 = st.columns(2)

        with col1:
            trajectories = st.number_input(
                "Количество траекторий Монте-Карло.", min_value=0, value=10000
            )
        with col2:
            seed = st.number_input(
                "Начальное значение для ГПСЧ", min_value=0, value=None
            )

        time_series_params = TimeSeriesBaseSystemParams()

        use_timeseries = st.checkbox("Использовать таймсериес", value=True)
        if use_timeseries:
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
                lambda_times = parse_1d_array(lambda_times_text, "lambda_times")
                lambda_values = parse_lambda_array(lambda_values_text, "lambda_values")
                mu_times = parse_1d_array(mu_times_text, "mu_times")
                mu_values = parse_1d_array(mu_values_text, "mu_values")
                nu_times = parse_1d_array(nu_times_text, "nu_times")
                nu_values = parse_1d_array(nu_values_text, "nu_values")

                time_series_params.lambda_rate = TimeSeries(lambda_times, lambda_values)
                time_series_params.mu_rate = TimeSeries(mu_times, mu_values)
                time_series_params.nu_rate = TimeSeries(nu_times, nu_values)

            except ValueError as e:
                st.error(str(e))
                st.stop()

        simulation_params = SimulationParams(trajectories, seed)

        params = ImitationSystemParams(
            base_params=base_params,
            transient_params=transient_params,
            calculation_params=settings,
            simulation_params=simulation_params,
            time_series_params=time_series_params,
        )

    return config, params


def _get_capacity_and_map_params(
    system_type: SystemType,
) -> tuple[int, int | None, np.ndarray | None, np.ndarray | None, int | None]:
    """Считывает параметры емкости системы и, при необходимости, параметры MAP-процесса.

    Args:
        system_type (SystemType): Тип выбранной системы.

    Returns:
        tuple[int, int | None, np.ndarray | None, np.ndarray | None, int | None]:
            - max_customers (int): Максимальное число заявок в системе.
            - processor_count (int | None): Число обслуживающих приборов, если применимо.
            - p_rate (np.ndarray | None): Матрица интенсивностей P для MAP-процесса.
            - q_rate (np.ndarray | None): Матрица интенсивностей Q для MAP-процесса.
            - sensor_count (int | None): Число датчиков для MAP, если применимо.
    """
    p_rate, q_rate = None, None
    sensor_count = None
    processor_count = None

    if system_type == SystemType.MAP:
        max_customers = (
            st.number_input(
                "Максимальное количество заявок в системе (n)",
                min_value=1,
                max_value=100,
                value=3,
            )
            + 2
        )

        sensor_count = st.number_input(
            "Число датчиков для MAP-процесса (m)",
            min_value=1,
            max_value=100,
            value=3,
        )

        st.markdown("---")

        p_rate, q_rate = map_intensity_matrix(sensor_count)
    else:
        max_customers = st.number_input(
            "Максимальное количество заявок в системе (n)",
            min_value=1,
            max_value=100,
            value=4,
        )

        processor_count = st.number_input(
            "Количество обслуживающих процессоров (m)",
            min_value=1,
            max_value=100,
            value=2,
        )

    return max_customers, processor_count, p_rate, q_rate, sensor_count


def _calculate_state_count(
    system_type: SystemType,
    max_customers: int,
    processor_count: int | None,
    sensor_count: int | None,
) -> int:
    """Вычисляет число состояний системы для задания начальных условий.

    Args:
        system_type (SystemType): Тип выбранной системы.
        max_customers (int): Максимальное число заявок в системе.
        processor_count (int | None): Число обслуживающих приборов.
        sensor_count (int | None): Число датчиков для MAP.

    Returns:
        int: Общее число состояний системы.
    """
    count = max_customers

    if system_type == SystemType.MULTI and processor_count is not None:
        count += processor_count + 1
    elif system_type == SystemType.MAP and sensor_count is not None:
        count *= sensor_count

    return count


def _get_transient_params(
    system_mode: SystemMode,
    state_count: int,
) -> TransientSystemParams | None:
    """Собирает параметры переходного режима, если выбран transient-режим.

    Args:
        system_mode (SystemMode): Режим работы системы.
        state_count (int): Число состояний системы.

    Returns:
        TransientSystemParams | None:
            Параметры переходного режима, если выбран режим TRANSIENT.
            В противном случае — None.
    """
    if system_mode != SystemMode.TRANSIENT:
        return None

    st.markdown("---")
    state_variables, initial_probabilities = render_initial_conditions(state_count)

    st.markdown("---")
    time_array = render_time_settings()

    return TransientSystemParams(
        time_array=time_array,
        state_variables=state_variables,
        initial_probabilities=initial_probabilities,
    )


def _build_base_params(
    system_type: SystemType,
    lambda_rate: float,
    mu_rate: float,
    nu_rate: float,
    max_customers: int,
    processor_count: int | None,
    sensor_count: int | None,
    p_rate: np.ndarray | None,
    q_rate: np.ndarray | None,
) -> BaseSystemParams | MAPSystemParams:
    """Формирует базовые параметры системы и расширяет их для MAP-моделей.

    Args:
        system_type (SystemType): Тип выбранной системы.
        lambda_rate (float): Интенсивность входного потока.
        mu_rate (float): Интенсивность обслуживания.
        nu_rate (float): Дополнительная интенсивность системы.
        max_customers (int): Максимальное число заявок в системе.
        sensor_count (int): Число датчиков для MAP-процесса.
        processor_count (int | None): Число обслуживающих приборов.
        p_rate (np.ndarray | None): Матрица интенсивностей P для MAP-процесса.
        q_rate (np.ndarray | None): Матрица интенсивностей Q для MAP-процесса.

    Returns:
        BaseSystemParams | MAPSystemParams:
            Базовые параметры системы. Для MAP-типов возвращается объект
            MAPSystemParams с матрицами p_rate и q_rate.
    """
    base_params = BaseSystemParams(
        mu_rate=mu_rate,
        nu_rate=nu_rate,
        lambda_rate=lambda_rate,
        max_customers=max_customers,
    )

    if system_type == SystemType.MULTI:
        return MultiSystemParams(
            **asdict(base_params),
            processor_count=processor_count,
        )

    if system_type == SystemType.MAP and p_rate is not None and q_rate is not None:
        return MAPSystemParams(
            **asdict(base_params),
            p_rate=p_rate.astype(np.float64),
            q_rate=q_rate.astype(np.float64),
            sensor_count=sensor_count,
        )

    return base_params
