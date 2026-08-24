"""Модуль визуализации вероятностей и пропускной способности состояний системы.

Содержит функции для построения графиков с использованием Plotly:
- Вероятности состояний системы во времени.
- Пропускная способность при разных параметрах нетерпения.
"""

from typing import Any

import numpy as np
import plotly.express as px  # type: ignore[import-untyped]
import plotly.graph_objs as go  # type: ignore[import-untyped]
from numpy.typing import NDArray


def plot_probabilities(
    p_matrix: NDArray[np.float64], time_array: NDArray[np.float64], title_text: str
) -> Any:
    """Создает график вероятностей по состояниям системы во времени.

    Args:
        p_matrix (NDArray[np.float64]): Матрица вероятностей.
        time_array (NDArray[np.float64]): Массив времени, соответствующий оси X.
        title_text (str): Текст заголовка графика.

    Returns:
        go.Figure: Объект графика Plotly с отображением вероятностей и их суммы.
    """
    fig = go.Figure()

    colors = px.colors.qualitative.D3

    for state in range(p_matrix.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=time_array,
                y=p_matrix[state],
                mode="lines",
                name=f"State {state}",
                line=dict(width=2, color=colors[state % len(colors)]),
            )
        )

    total_prob = p_matrix.sum(axis=0)
    fig.add_trace(
        go.Scatter(
            x=time_array,
            y=total_prob,
            mode="lines",
            name="Total probability",
            line=dict(width=2, dash="dash", color="black"),
        )
    )

    annotations = []
    if not np.allclose(total_prob, 1.0, atol=0.01):
        annotations.append(
            dict(
                x=0.05,
                y=0.95,
                xref="paper",
                yref="paper",
                text="Внимание: Суммарная вероятность ≠ 1!",
                showarrow=False,
                font=dict(color="red", size=14),
            )
        )

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=20, color="black"),
        ),
        xaxis=dict(
            title="Time (t)",
            title_font=dict(color="black", size=16),
            tickfont=dict(color="black"),
            automargin=True,
            range=[0, time_array[-1]],
            constrain="domain",
        ),
        yaxis=dict(
            title="Probability",
            title_font=dict(color="black", size=16),
            tickfont=dict(color="black"),
            automargin=True,
            range=[0, 1.0],
            constrain="domain",
        ),
        width=1152,
        height=768,
        plot_bgcolor="white",
        paper_bgcolor="white",
        template="plotly_white",
        font=dict(family="Arial", size=14, color="black"),
        legend=dict(bordercolor="LightGray", borderwidth=1, font=dict(color="black")),
        annotations=annotations,
        margin=dict(l=60, r=40, t=80, b=60),
    )

    fig.update_xaxes(showgrid=True, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridcolor="LightGray", range=[0, 1.05])

    return fig


def plot_metric(
    metric_data: list[NDArray[np.float64]] | NDArray[np.float64],
    time_array: NDArray[np.float64],
    title_text: str,
    yaxis_title: str = "Value",
    series_labels: list[str] | None = None,
    line_colors: list[str] | None = None,
    reference_lines: list[dict[str, Any]] | None = None,
) -> Any:
    """Универсальная функция построения графика для различных метрик системы.

    Args:
        metric_data (list[NDArray[np.float64]] | NDArray[np.float64]):
            Список массивов метрик или одиночный массив (если один график).
        time_array (NDArray[np.float64]): Массив значений времени.
        title_text (str): Заголовок графика.
        yaxis_title (str): Подпись оси Y (по умолчанию "Value").
        series_labels (list[str] | None): Пользовательские подписи для каждой серии.
        line_colors (list[str] | None): Цвета линий (по умолчанию из Plotly D3 палитры).
        reference_lines (list[dict[str, Any]] | None): Опорные линии вида
            ``{"x": [...], "y": [...], "name": str}``, наносимые поверх
            метрики: например, критический уровень характеристики.

    Returns:
        go.Figure: Объект графика Plotly.
    """
    fig = go.Figure()
    default_colors = px.colors.qualitative.D3

    metric_arr = (
        np.asarray(metric_data) if not isinstance(metric_data, list) else metric_data
    )

    if isinstance(metric_arr, list):
        metric_data = metric_arr
    elif metric_arr.ndim == 1:
        metric_data = [metric_arr]
    elif metric_arr.ndim == 2:
        metric_data = [metric_arr[:, i] for i in range(metric_arr.shape[1])]

    if series_labels is None:
        series_labels = [f"State {i}" for i in range(len(metric_data))]

    if line_colors is None:
        line_colors = default_colors

    for idx, data in enumerate(metric_data):
        label = series_labels[idx] if idx < len(series_labels) else f"Series {idx}"
        color = line_colors[idx % len(line_colors)]
        fig.add_trace(
            go.Scatter(
                x=time_array,
                y=data,
                mode="lines",
                name=label,
                line=dict(width=2, color=color),
            )
        )

    for line in reference_lines or []:
        fig.add_trace(
            go.Scatter(
                x=line["x"],
                y=line["y"],
                mode="lines",
                name=line.get("name", "Опорный уровень"),
                line=dict(width=2, dash="dash", color=line.get("color", "#d62728")),
                hoverinfo="name+y",
            )
        )

    if len(metric_data) > 1:
        total_series = np.sum(metric_data, axis=0)
        fig.add_trace(
            go.Scatter(
                x=time_array,
                y=total_series,
                mode="lines",
                name="Total",
                line=dict(width=2, dash="dash", color="black"),
            )
        )

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=20, color="black"),
        ),
        xaxis=dict(
            title="Time (t)",
            title_font=dict(color="black", size=16),
            tickfont=dict(color="black"),
            automargin=True,
            range=[0, time_array[-1]],
            constrain="domain",
        ),
        yaxis=dict(
            title=yaxis_title,
            title_font=dict(color="black", size=16),
            tickfont=dict(color="black"),
            automargin=True,
            constrain="domain",
        ),
        width=1152,
        height=768,
        plot_bgcolor="white",
        paper_bgcolor="white",
        template="plotly_white",
        font=dict(family="Arial", size=14, color="black"),
        legend=dict(bordercolor="LightGray", borderwidth=1, font=dict(color="black")),
        margin=dict(l=60, r=40, t=80, b=60),
    )

    fig.update_xaxes(showgrid=True, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridcolor="LightGray")

    return fig
