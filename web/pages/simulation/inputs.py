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
    system_capacity_inputs,
)

from quick.domain import ComputationConfig, SystemMode, SystemType
from quick.domain.models import (
    BaseSystemParams,
    CalculationSettings,
    MAPSystemParams,
    SystemParams,
    TransientSystemParams,
)


def get_user_inputs() -> tuple[ComputationConfig, SystemParams]:
    """Собирает все входные параметры от пользователя через UI.

    Returns:
        tuple[ComputationConfig, SystemParams]:
            - config (ComputationConfig): Конфигурация выполнения вычислений.
            - params (SystemParams): Параметры выбранной СМО.
    """
    config = render_calculation_config()
    st.markdown("---")

    system_type, calculation_mode, system_mode = get_system_mode_type()
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
        p_rate=p_rate,
        q_rate=q_rate,
    )

    settings = CalculationSettings(
        calculation_mode=calculation_mode,
        system_mode=system_mode,
        system_type=system_type,
    )

    params = SystemParams(
        base_params=base_params,
        transient_params=transient_params,
        settings=settings,
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
            - sensor_count (int | None): Число датчиков для MULTI_SENSOR_MAP, если применимо.
    """
    p_rate, q_rate = None, None
    sensor_count = None

    if system_type in (SystemType.MAP, SystemType.MULTI_SENSOR_MAP):
        max_customers, processor_count = system_capacity_inputs(
            system_type,
            default_max_customers=3,
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

        map_dimension = max_customers if system_type == SystemType.MAP else sensor_count
        p_rate, q_rate = map_intensity_matrix(map_dimension)
    else:
        max_customers, processor_count = system_capacity_inputs(system_type)

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
        sensor_count (int | None): Число датчиков для MULTI_SENSOR_MAP.

    Returns:
        int: Общее число состояний системы.
    """
    count = max_customers

    if system_type == SystemType.MULTI and processor_count is not None:
        count += processor_count + 1
    elif system_type == SystemType.MAP:
        count **= 2
    elif system_type == SystemType.MULTI_SENSOR_MAP and sensor_count is not None:
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
        processor_count=processor_count,
    )

    if (
        system_type in (SystemType.MAP, SystemType.MULTI_SENSOR_MAP)
        and p_rate is not None
        and q_rate is not None
    ):
        return MAPSystemParams(
            **asdict(base_params),
            p_rate=p_rate.astype(np.float64),
            q_rate=q_rate.astype(np.float64),
        )

    return base_params
