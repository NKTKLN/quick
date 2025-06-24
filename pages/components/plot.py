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
    p_matrix: NDArray[np.float64], time_array: NDArray[np.float64]
) -> Any:
    """Создает график вероятностей по состояниям системы во времени.

    Args:
        p_matrix (NDArray[np.float64]): Матрица вероятностей.
        time_array (NDArray[np.float64]): Массив времени, соответствующий оси X.

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
            text="System States Probabilities", font=dict(size=20, color="black")
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


def plot_throughput(
    throughput_results: list[NDArray[np.float64]] | NDArray[np.float64],
    time_array: NDArray[np.float64],
) -> Any:
    """Создает график пропускной способности системы по состояниям.

    Args:
        throughput_results (list[NDArray[np.float64]] | NDArray[np.float64]):
            Список массивов пропускной способности для каждого состояния.
        time_array (NDArray[np.float64]): Массив значений параметра.

    Returns:
        go.Figure: Объект графика Plotly с линиями пропускной способности.
    """
    fig = go.Figure()

    colors = px.colors.qualitative.D3

    if not isinstance(throughput_results, list):
        throughput_results = [throughput_results]

    for state, throughput in enumerate(throughput_results):
        fig.add_trace(
            go.Scatter(
                x=time_array,
                y=throughput,
                mode="lines",
                name=f"State {state}",
                line=dict(width=2, color=colors[state % len(colors)]),
            )
        )

    fig.update_layout(
        title=dict(
            text="System Throughput",
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
            title="Throughput",
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
