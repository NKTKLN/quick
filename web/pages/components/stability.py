"""Компоненты настройки и отображения оценки устойчивости СМО.

Устойчивость здесь — это способность системы удерживать качество
обслуживания в переходном режиме: базовая характеристика ``a(t)`` не
должна опускаться ниже критического уровня ``a_кр``. Модуль отвечает за
ввод параметров этой оценки и за вывод её результата.
"""

from typing import Any

import numpy as np
import streamlit as st
from numpy.typing import NDArray

from quick.domain import StabilityMetric, StabilityVerdict
from quick.domain.params.stability import (
    BORDERLINE_THRESHOLD,
    DEFAULT_CRITICAL_SERVICE_PROBABILITY,
    DEFAULT_SETTLING_TOLERANCE,
    DEFAULT_STABLE_THRESHOLD,
    StabilityParams,
)
from quick.services.stability import StabilityResult

STABILITY_GUIDE = """
Система считается **устойчивой**, если базовая характеристика `a(t)` на
всём переходном интервале не опускается ниже критического уровня `a_кр`.
Это не классическое условие `ρ < 1`: СМО с конечным буфером и
нетерпеливыми заявками стационарна при любых интенсивностях, но в
переходном режиме способна нарушить требования к качеству обслуживания.

| Величина | Смысл |
| --- | --- |
| `K_уст(t) = a(t) / a_кр` | Мгновенный коэффициент: ниже единицы — требование нарушено |
| `K_уст = ∫a(t)dt / (a_кр · t_пер)` | Интегральный коэффициент за переходный процесс |
| `R, %` | Запас: площадь между `a(t)` и `a_кр`, нормированная на `(a(t₀) − a_кр)·t_пер` |
| `t_пер` | Длительность переходного процесса |

Показатель `R` взят из ВКР без изменений. Его знаменатель считает
наибольшим значением характеристики начальное, что верно для спадающего
переходного процесса. Расчёт с пустой системы даёт растущую `a(t)`, и
тогда `R` выходит за 100%: сравнивать варианты системы между собой в этом
случае нужно по `K_уст`.
"""


def render_stability_params(key_prefix: str = "stability") -> StabilityParams:
    """Отображает UI настройки оценки устойчивости.

    Args:
        key_prefix (str): Префикс ключей Streamlit-виджетов.

    Returns:
        StabilityParams: Параметры оценки устойчивости.
    """
    st.subheader("🛡️ Устойчивость системы")

    with st.expander("📖 Как определяется устойчивость", expanded=False):
        st.markdown(STABILITY_GUIDE)

    metric = st.selectbox(
        "Характеристика оценки устойчивости a(t)",
        list(StabilityMetric),
        format_func=lambda item: item.value,
        key=f"{key_prefix}_metric",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        # Для безразмерной вероятности обслуживания критический уровень
        # ограничен единицей, для пропускной способности — нет: он задаётся
        # в тех же единицах, что и сама A(t).
        if metric == StabilityMetric.SERVICE_PROBABILITY:
            critical_level = st.number_input(
                "Критический уровень a_кр",
                min_value=0.0,
                max_value=0.999999,
                value=DEFAULT_CRITICAL_SERVICE_PROBABILITY,
                format="%.6f",
                key=f"{key_prefix}_critical_probability",
                help="1 − допустимая доля потерь. 0,95 соответствует 5% потерь.",
            )
        else:
            critical_level = st.number_input(
                "Критический уровень A_crit, заявки/ед. времени",
                min_value=0.0,
                value=100.0,
                format="%.6f",
                key=f"{key_prefix}_critical_throughput",
                help="Минимально допустимый поток обслуженных заявок.",
            )

    with col2:
        stable_threshold = st.number_input(
            "Порог устойчивости K_уст",
            min_value=BORDERLINE_THRESHOLD,
            value=DEFAULT_STABLE_THRESHOLD,
            step=0.05,
            format="%.2f",
            key=f"{key_prefix}_threshold",
            help="Значение K_уст, начиная с которого запас считается достаточным.",
        )

    with col3:
        settling_tolerance = st.number_input(
            "Допуск выхода на режим",
            min_value=0.001,
            max_value=0.999,
            value=DEFAULT_SETTLING_TOLERANCE,
            step=0.01,
            format="%.3f",
            key=f"{key_prefix}_tolerance",
            help="Ширина коридора вокруг установившегося значения для оценки t_пер.",
        )

    return StabilityParams(
        metric=metric,
        critical_level=float(critical_level),
        settling_tolerance=float(settling_tolerance),
        stable_threshold=float(stable_threshold),
    )


def render_stability_summary(result: StabilityResult, params: StabilityParams) -> None:
    """Отображает итоговую оценку устойчивости системы.

    Args:
        result (StabilityResult): Результат оценки устойчивости.
        params (StabilityParams): Параметры оценки.
    """
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("K_уст (интегральный)", f"{result.coefficient:.4f}")
    col2.metric("Запас R", f"{result.margin:.2f} %")
    col3.metric("t_пер", f"{result.settling_time:.6g}")
    # Иконку в плитку не ставим: вместе с ней «Неустойчивый» не помещается
    # в колонку и обрезается. Цветное сообщение ниже несёт её и без того.
    col4.metric("Режим", result.verdict.value)

    # Формула R из ВКР нормирована по начальному значению характеристики.
    # Для растущего переходного процесса оно не является наибольшим, и
    # показатель выходит за 100% — об этом нужно предупредить, иначе
    # значение читается как ошибка расчёта.
    # В расчёте по одному датчику характеристика не определена там, где
    # этот датчик не порождает заявок: делить поток ухода не на что.
    if result.skipped_points:
        st.caption(
            f"ℹ️ Характеристика не определена в {result.skipped_points} "
            "временных точках — там выбранный датчик не порождает заявок. "
            "Эти точки исключены из оценки."
        )

    if result.margin > 100.0:
        st.caption(
            "ℹ️ R > 100%: характеристика поднимается выше своего начального "
            "значения, по которому формула ВКР нормирует запас. Для такого "
            "переходного процесса сравнивайте варианты системы по K_уст."
        )

    boundaries = (
        f"устойчивый — K_уст ≥ {params.stable_threshold:.2f}; "
        f"пограничный — {BORDERLINE_THRESHOLD:.2f} ≤ K_уст < "
        f"{params.stable_threshold:.2f}; "
        f"неустойчивый — K_уст < {BORDERLINE_THRESHOLD:.2f} либо "
        "просадка ниже критического уровня."
    )

    if result.verdict == StabilityVerdict.UNSTABLE:
        st.error(
            f"❌ Система неустойчива: минимум a(t) = {result.minimum:.6g} "
            f"при a_кр = {result.critical_level:.6g}. Границы: {boundaries}"
        )
    elif result.verdict == StabilityVerdict.BORDERLINE:
        st.warning(
            f"⚠️ Пограничный режим: требование выполняется, но запас меньше "
            f"заданного порога. Границы: {boundaries}"
        )
    else:
        st.success(f"✅ Система устойчива с запасом. Границы: {boundaries}")


def stability_reference_lines(
    time_array: NDArray[np.float64],
) -> list[dict[str, Any]]:
    """Формирует описание горизонтали K_уст = 1 для двумерного графика.

    Единица — граница, ниже которой характеристика не выдерживает
    критического уровня, поэтому на графике коэффициента она нужна как
    точка отсчёта.

    Args:
        time_array (NDArray[np.float64]): Массив времени.

    Returns:
        list[dict[str, Any]]: Описание линии для Plotly.
    """
    return [
        dict(
            x=[float(time_array[0]), float(time_array[-1])],
            y=[BORDERLINE_THRESHOLD, BORDERLINE_THRESHOLD],
            name="Критический уровень",
        )
    ]
