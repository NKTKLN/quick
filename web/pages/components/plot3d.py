"""Модуль построения трёхмерных графиков характеристик СМО.

Содержит функции для визуализации метрик системы в трёх измерениях:
- поверхность «время × параметр × метрика» — для развёртки по параметру;
- поверхность «время × датчик × метрика» — для покомпонентного расчёта;
- фазовый портрет «метрика × метрика × время» — для одного расчёта.

Все функции возвращают объект Plotly и оформляются единообразно с
двумерными графиками из модуля plot.
"""

from typing import Any

import numpy as np
import plotly.graph_objs as go  # type: ignore[import-untyped]
from numpy.typing import NDArray

PLOT_WIDTH = 1152
PLOT_HEIGHT = 768
TIME_AXIS_TITLE = "Time (t)"


def _axis_settings(title: str, **extra: Any) -> dict[str, Any]:
    """Формирует настройки оси трёхмерной сцены.

    Args:
        title (str): Подпись оси.
        **extra (Any): Дополнительные параметры оси Plotly.

    Returns:
        dict[str, Any]: Словарь настроек оси.
    """
    return dict(
        title=dict(text=title, font=dict(color="black", size=14)),
        tickfont=dict(color="black", size=11),
        backgroundcolor="white",
        gridcolor="LightGray",
        showbackground=True,
        zerolinecolor="Gray",
        **extra,
    )


def _apply_layout(
    fig: Any,
    title_text: str,
    xaxis_title: str,
    yaxis_title: str,
    zaxis_title: str,
    yaxis_extra: dict[str, Any] | None = None,
) -> Any:
    """Применяет общее оформление к трёхмерному графику.

    Args:
        fig (go.Figure): Объект графика Plotly.
        title_text (str): Заголовок графика.
        xaxis_title (str): Подпись оси X.
        yaxis_title (str): Подпись оси Y (третья ось графика).
        zaxis_title (str): Подпись оси Z.
        yaxis_extra (dict[str, Any] | None): Дополнительные настройки оси Y.

    Returns:
        go.Figure: Объект графика с применённым оформлением.
    """
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=20, color="black")),
        scene=dict(
            xaxis=_axis_settings(xaxis_title),
            yaxis=_axis_settings(yaxis_title, **(yaxis_extra or {})),
            zaxis=_axis_settings(zaxis_title),
            camera=dict(eye=dict(x=1.7, y=-1.7, z=0.9)),
        ),
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        template="plotly_white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=14, color="black"),
        margin=dict(l=40, r=40, t=80, b=40),
    )
    return fig


def plot_surface(
    values: NDArray[np.float64],
    time_array: NDArray[np.float64],
    axis_values: NDArray[np.float64],
    title_text: str,
    yaxis_title: str,
    zaxis_title: str = "Value",
    colorscale: str = "Viridis",
    yaxis_extra: dict[str, Any] | None = None,
) -> Any:
    """Строит поверхность «время × третья ось × метрика».

    Args:
        values (NDArray[np.float64]): Значения метрики формы
            [len(axis_values), len(time_array)].
        time_array (NDArray[np.float64]): Массив времени (ось X).
        axis_values (NDArray[np.float64]): Значения третьей оси (ось Y).
        title_text (str): Заголовок графика.
        yaxis_title (str): Подпись третьей оси.
        zaxis_title (str): Подпись оси значений метрики.
        colorscale (str): Цветовая шкала поверхности.
        yaxis_extra (dict[str, Any] | None): Дополнительные настройки оси Y.

    Returns:
        go.Figure: Объект графика Plotly с поверхностью.

    Raises:
        ValueError: Если форма values не соответствует осям X и Y.
    """
    values = np.asarray(values, dtype=np.float64)
    expected_shape = (len(axis_values), len(time_array))

    if values.shape != expected_shape:
        raise ValueError(
            "Форма значений метрики не соответствует осям графика: "
            f"{values.shape} вместо {expected_shape}."
        )

    fig = go.Figure(
        data=go.Surface(
            x=time_array,
            y=axis_values,
            z=values,
            colorscale=colorscale,
            colorbar=dict(title=dict(text=zaxis_title, font=dict(color="black"))),
        )
    )

    return _apply_layout(
        fig,
        title_text=title_text,
        xaxis_title=TIME_AXIS_TITLE,
        yaxis_title=yaxis_title,
        zaxis_title=zaxis_title,
        yaxis_extra=yaxis_extra,
    )


def plot_sensor_surface(
    values: NDArray[np.float64],
    time_array: NDArray[np.float64],
    title_text: str,
    zaxis_title: str = "Value",
) -> Any:
    """Строит поверхность «время × номер датчика × метрика».

    Args:
        values (NDArray[np.float64]): Значения метрики формы
            [len(time_array), sensor_count].
        time_array (NDArray[np.float64]): Массив времени (ось X).
        title_text (str): Заголовок графика.
        zaxis_title (str): Подпись оси значений метрики.

    Returns:
        go.Figure: Объект графика Plotly с поверхностью.
    """
    values = np.asarray(values, dtype=np.float64)
    sensor_indices = np.arange(values.shape[1], dtype=np.float64)

    return plot_surface(
        values=values.T,
        time_array=time_array,
        axis_values=sensor_indices,
        title_text=title_text,
        yaxis_title="Номер датчика",
        zaxis_title=zaxis_title,
        colorscale="Turbo",
        yaxis_extra=dict(tickmode="array", tickvals=sensor_indices),
    )


def plot_phase_portrait(
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
    time_array: NDArray[np.float64],
    title_text: str,
    xaxis_title: str,
    yaxis_title: str,
) -> Any:
    """Строит фазовый портрет «метрика × метрика × время».

    Траектория окрашивается по времени, что позволяет увидеть направление
    движения системы и её выход на стационарный режим.

    Args:
        x_values (NDArray[np.float64]): Значения первой метрики (ось X).
        y_values (NDArray[np.float64]): Значения второй метрики (ось Y).
        time_array (NDArray[np.float64]): Массив времени (ось Z).
        title_text (str): Заголовок графика.
        xaxis_title (str): Подпись оси первой метрики.
        yaxis_title (str): Подпись оси второй метрики.

    Returns:
        go.Figure: Объект графика Plotly с траекторией системы.
    """
    fig = go.Figure(
        data=go.Scatter3d(
            x=x_values,
            y=y_values,
            z=time_array,
            mode="lines",
            line=dict(
                width=6,
                color=time_array,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title=dict(text=TIME_AXIS_TITLE, font=dict(color="black"))
                ),
            ),
            name="Траектория",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[x_values[0], x_values[-1]],
            y=[y_values[0], y_values[-1]],
            z=[time_array[0], time_array[-1]],
            mode="markers+text",
            marker=dict(size=6, color=["#2ca02c", "#d62728"]),
            text=["старт", "финиш"],
            textposition="top center",
            showlegend=False,
        )
    )

    return _apply_layout(
        fig,
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        zaxis_title=TIME_AXIS_TITLE,
    )
