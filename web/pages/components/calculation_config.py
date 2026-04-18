"""Компонент интерфейса Streamlit для настройки параметров вычислений.

Позволяет выбрать тип вычислений (NumPy, Mpmath или объединённый), настроить точность и
погрешность, а также включить или отключить кэширование.
"""

import streamlit as st

from quick.domain import (
    CalculationEngine,
    CalculationMethod,
    ComputationConfig,
    MergedComputationConfig,
    MpmathComputationConfig,
)
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
            [method.value for method in CalculationMethod],
        )
    )
    if calculation_method != CalculationMethod.ANALYTICAL:
        return ComputationConfig(
            calculation_engine=CalculationEngine.NUMPY,
            calculation_method=calculation_method,
        )

    calculation_type = st.selectbox(
        "Выберите тип вычисления:",
        [
            "Numpy (быстрое, для небольших систем)",
            "Mpmath (точное, для больших систем)",
            "Объединенное (для сложных систем с высокой точностью)",
        ],
        index=2,
    )

    precision = 50
    if calculation_type in [
        "Mpmath (точное, для больших систем)",
        "Объединенное (для сложных систем с высокой точностью)",
    ]:
        precision = st.slider(
            "Настройка точности вычислений (колличество знаков после запятой):",
            min_value=16,
            max_value=256,
            step=1,
            value=precision,
        )

    tolerance = precision
    if calculation_type == "Объединенное (для сложных систем с высокой точностью)":
        tolerance = st.slider(
            "Настройка допустимой погрешности (колличество знаков после запятой):",
            min_value=1,
            max_value=256,
            step=1,
            value=precision,
        )

    if calculation_type == "Numpy (быстрое, для небольших систем)":
        config = ComputationConfig(
            calculation_engine=CalculationEngine.NUMPY,
            calculation_method=CalculationMethod(calculation_method),
        )
    elif calculation_type == "Mpmath (точное, для больших систем)":
        config = MpmathComputationConfig(
            calculation_engine=CalculationEngine.MPMATH,
            calculation_method=CalculationMethod(calculation_method),
        )
        config.precision = precision
    else:
        config = MergedComputationConfig(
            calculation_engine=CalculationEngine.MERGED,
            calculation_method=CalculationMethod(calculation_method),
        )
        config.precision = precision
        config.tolerance = 10 ** (-tolerance)

    app_config = ConfigLoader.get_config()
    if not app_config.disable_cache:
        disable_cache = st.checkbox("Выключить кэширование", value=config.disable_cache)

        if disable_cache != config.disable_cache:
            config.disable_cache = disable_cache

    return config
