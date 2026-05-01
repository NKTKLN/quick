"""Компонент интерфейса Streamlit для настройки параметров вычислений."""

import streamlit as st

from quick.domain import (
    CalculationEngine,
    CalculationMethod,
    ComputationConfig,
)
from quick.domain.models import SimulationParams
from quick.settings import ConfigLoader


def render_calculation_config() -> ComputationConfig:
    """Отображает UI-компонент для конфигурации вычислений.

    В зависимости от выбранного типа вычислений, позволяет настраивать точность
    и погрешность. Также предоставляет возможность включать или выключать кэш.

    Returns:
        ComputationConfig: Объект конфигурации.
    """
    st.subheader("⚙️ Настройки вычисления для системы")
    calculation_method = CalculationMethod(
        st.selectbox(
            "Выберите метод вычисления:",
            [CalculationMethod.IMITATION.value],
            disabled=True
        )
    )

    calculation_type = CalculationEngine.NUMPY

    if calculation_type == CalculationEngine.NUMPY:
        config = ComputationConfig(
            calculation_engine=CalculationEngine(calculation_type),
            calculation_method=CalculationMethod(calculation_method),
        )

    app_config = ConfigLoader.get_config()
    if not app_config.disable_cache:
        disable_cache = st.checkbox("Выключить кэширование", value=config.disable_cache)

        if disable_cache != config.disable_cache:
            config.disable_cache = disable_cache

    return config


def render_imitation_config() -> SimulationParams:
    """Отображает UI-компоненты для задания параметров имитационного моделирования.

    Returns:
        SimulationParams: Объект с параметрами имитационного моделирования,
            содержащий число траекторий и seed генератора случайных чисел.
    """
    col1, col2 = st.columns(2)

    with col1:
        trajectories = st.number_input(
            "Количество траекторий Монте-Карло.", min_value=0, value=10000
        )
    with col2:
        seed = st.number_input("Начальное значение для ГПСЧ", min_value=0, value=None)

    simulation_params = SimulationParams(trajectories, seed)

    return simulation_params
