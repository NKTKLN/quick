"""Модуль для визуализации характеристик системы массового обслуживания (СМО).

Содержит функции для графического представления динамики:
- Вероятностей состояний системы (plot_probabilities)
- Пропускной способности системы (plot_throughput)

Основные возможности модуля:
1. Визуализация:
   - Графики вероятностей всех состояний системы
   - Суммарной вероятности состояний (контроль корректности)
   - Динамики пропускной способности при различных параметрах
2. Гибкая настройка:
   - Поддержка различных входных форматов данных
   - Автоматическая настройка осей и легенды
   - Кастомизация стилей графиков
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_probabilities(p_matrix: np.ndarray, time_array: np.ndarray) -> plt.Figure:
    """Строит график вероятностей состояний системы и их суммарной вероятности.

    Args:
        p_matrix: Матрица вероятностей размером (n_states, n_timesteps)
        time_array: Массив временных меток размером (n_timesteps,)

    Returns:
        matplotlib.Figure: Объект с построенным графиком
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Графики вероятностей для каждого состояния
    for state in range(p_matrix.shape[0]):
        ax.plot(time_array, p_matrix[state], linewidth=2)

    # График суммарной вероятности
    total_prob = p_matrix.sum(axis=0)
    ax.plot(time_array, total_prob, "k--", label="Total probability", linewidth=2)

    _configure_plot(ax, "System States Probabilities", "Time (t)", "Probability")

    if not np.allclose(total_prob, 1.0, atol=0.01):
        ax.text(
            0.05,
            0.95,
            "Внимание: Суммарная вероятность ≠ 1!",
            transform=ax.transAxes,
            color="red",
            fontsize=12,
        )

    return fig


def plot_throughput(
    throughput_results: List[np.ndarray], time_array: np.ndarray
) -> plt.Figure:
    """Строит график пропускной способности.

    Args:
        throughput_results: Массив значений пропускной способности
        time_array: Массив временных меток размером (n_timesteps,)

    Returns:
        matplotlib.Figure: Объект с построенным графиком
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Графики пропускной способности для каждого состояния
    for state in range(len(throughput_results)):
        ax.plot(
            time_array, throughput_results[state], linewidth=2, label=f"State {state}"
        )

    _configure_plot(
        ax, "System Throughput vs Impatience Rate", "Impatience rate (ν)", "Throughput"
    )

    return fig


def _configure_plot(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Настраивает общие параметры графиков.

    Args:
        ax: Ось графика
        title: Заголовок графика
        xlabel: Подпись оси X
        ylabel: Подпись оси Y
    """
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(fontsize=12)
