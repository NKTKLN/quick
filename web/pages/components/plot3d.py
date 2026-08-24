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
TIME_AXIS_TITLE = "Время t"

# Оформление осей сцены. Все три оси используют одни и те же размеры
# подписей, одинаковые деления и одинаковую сетку, чтобы ни одна из них
# не выглядела главной: график читается как зависимость от двух
# равноправных параметров.
AXIS_TITLE_SIZE = 14
AXIS_TICK_SIZE = 11
AXIS_TICK_COUNT = 6
GRID_COLOR = "#B8BFC7"
GRID_WIDTH = 2
AXIS_LINE_COLOR = "#5A6472"

# Цветовая шкала намеренно узкая и короче области построения: на
# трёхмерном графике она несёт вспомогательную роль, а полноразмерная
# шкала отбирает у поверхности заметную долю ширины.
COLORBAR_LENGTH = 0.55
COLORBAR_THICKNESS = 12
COLORBAR_TICK_SIZE = 9
COLORBAR_TITLE_SIZE = 11
COLORBAR_X = 0.94
COLORBAR_Y = 0.5
SCENE_RIGHT_EDGE = 0.92
SCENE_BOTTOM_EDGE = 0.10

# Контурные линии на поверхности показывают области локальных максимумов
# и резких изменений характеристики. Цвет нейтральный и полупрозрачный,
# чтобы линии читались, но не спорили с самой поверхностью.
CONTOUR_COLOR = "rgba(25, 28, 35, 0.6)"
CONTOUR_WIDTH = 3
CONTOUR_COUNT = 12


def _axis_settings(title: str, **extra: Any) -> dict[str, Any]:
    """Формирует настройки оси трёхмерной сцены.

    Оформление одинаково для X, Y и Z: сетка включена по всем трём осям,
    размеры подписей и делений совпадают.

    Args:
        title (str): Подпись оси.
        **extra (Any): Дополнительные параметры оси Plotly.

    Returns:
        dict[str, Any]: Словарь настроек оси.
    """
    return dict(
        title=dict(text=title, font=dict(color="black", size=AXIS_TITLE_SIZE)),
        tickfont=dict(color="black", size=AXIS_TICK_SIZE),
        backgroundcolor="white",
        showbackground=True,
        showgrid=True,
        gridcolor=GRID_COLOR,
        gridwidth=GRID_WIDTH,
        zeroline=True,
        zerolinecolor=AXIS_LINE_COLOR,
        zerolinewidth=GRID_WIDTH,
        showline=True,
        linecolor=AXIS_LINE_COLOR,
        linewidth=GRID_WIDTH,
        ticks="outside",
        ticklen=4,
        tickwidth=1,
        tickcolor=AXIS_LINE_COLOR,
        nticks=AXIS_TICK_COUNT,
        **extra,
    )


def _colorbar_settings(title: str) -> dict[str, Any]:
    """Формирует компактную цветовую шкалу.

    Args:
        title (str): Заголовок шкалы.

    Returns:
        dict[str, Any]: Словарь настроек colorbar.
    """
    return dict(
        title=dict(
            text=title,
            font=dict(color="black", size=COLORBAR_TITLE_SIZE),
            side="right",
        ),
        tickfont=dict(color="black", size=COLORBAR_TICK_SIZE),
        len=COLORBAR_LENGTH,
        thickness=COLORBAR_THICKNESS,
        x=COLORBAR_X,
        y=COLORBAR_Y,
        xanchor="left",
        yanchor="middle",
        outlinewidth=0,
        ticks="outside",
        ticklen=3,
        nticks=6,
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
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
            # Куб уравнивает три оси и на графике: ни одна из них не
            # выглядит растянутой относительно остальных, независимо от
            # того, в каких единицах измерена характеристика.
            aspectmode="cube",
            # Сцена заканчивается левее цветовой шкалы, поэтому шкала
            # не накладывается на поверхность ни при каком повороте.
            # Нижний край приподнят: контейнер Streamlit ужимает график по
            # ширине, и подписи оси времени иначе срезаются краем области.
            domain=dict(x=[0.0, SCENE_RIGHT_EDGE], y=[SCENE_BOTTOM_EDGE, 1.0]),
        ),
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        template="plotly_white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=14, color="black"),
        margin=dict(l=10, r=10, t=60, b=40),
    )
    return fig


def _contour_settings(
    values: NDArray[np.float64], show_contours: bool
) -> dict[str, Any]:
    """Формирует настройки линий уровня на поверхности.

    Шаг линий задаётся явно от размаха значений: автоматический подбор
    Plotly на пологих поверхностях оставляет одну-две линии, по которым
    рельеф не читается.

    Args:
        values (NDArray[np.float64]): Значения метрики.
        show_contours (bool): Наносить ли линии уровня.

    Returns:
        dict[str, Any]: Словарь настроек contours для go.Surface.
    """
    settings: dict[str, Any] = dict(
        show=show_contours,
        color=CONTOUR_COLOR,
        width=CONTOUR_WIDTH,
        highlight=False,
    )

    finite_values = values[np.isfinite(values)]

    if show_contours and finite_values.size:
        minimum = float(np.min(finite_values))
        maximum = float(np.max(finite_values))
        span = maximum - minimum

        if span > 0:
            settings |= dict(
                start=minimum,
                end=maximum,
                size=span / CONTOUR_COUNT,
            )

    return dict(z=settings)


def plot_surface(
    values: NDArray[np.float64],
    time_array: NDArray[np.float64],
    axis_values: NDArray[np.float64],
    title_text: str,
    yaxis_title: str,
    zaxis_title: str = "Value",
    colorscale: str = "Viridis",
    yaxis_extra: dict[str, Any] | None = None,
    show_contours: bool = True,
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
        show_contours (bool): Наносить ли линии уровня на поверхность.

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
            colorbar=_colorbar_settings(zaxis_title),
            contours=_contour_settings(values, show_contours),
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
    show_contours: bool = True,
) -> Any:
    """Строит поверхность «время × номер датчика × метрика».

    Датчики нумеруются с единицы, как в интерфейсе выбора датчика.

    Args:
        values (NDArray[np.float64]): Значения метрики формы
            [len(time_array), sensor_count].
        time_array (NDArray[np.float64]): Массив времени (ось X).
        title_text (str): Заголовок графика.
        zaxis_title (str): Подпись оси значений метрики.
        show_contours (bool): Наносить ли линии уровня на поверхность.

    Returns:
        go.Figure: Объект графика Plotly с поверхностью.
    """
    values = np.asarray(values, dtype=np.float64)
    sensor_numbers = np.arange(1, values.shape[1] + 1, dtype=np.float64)

    return plot_surface(
        values=values.T,
        time_array=time_array,
        axis_values=sensor_numbers,
        title_text=title_text,
        yaxis_title="Номер датчика",
        zaxis_title=zaxis_title,
        colorscale="Turbo",
        yaxis_extra=dict(tickmode="array", tickvals=sensor_numbers),
        show_contours=show_contours,
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
                colorbar=_colorbar_settings(TIME_AXIS_TITLE),
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
