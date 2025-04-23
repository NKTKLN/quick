"""Модуль для визуализации характеристик системы массового обслуживания (СМО).

Содержит функции для графического представления:
1. plot_probabilities - отображает вероятности состояний системы во времени
2. plot_throughput - визуализирует пропускную способность системы

Основные возможности:
- Построение графиков вероятностей всех состояний системы
- Отображение суммарной вероятности состояний
- Визуализация динамики пропускной способности
- Гибкие настройки оформления графиков
- Возможность сохранения результатов в файл
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_probabilities(
    p_matrix: np.ndarray,
    time_array: np.ndarray,
    save_to_file: bool = False,
    filename: str = "probabilities_plot.png"
) -> None:
    """Строит график вероятностей состояний системы и их суммарной вероятности.

    Args:
        p_matrix: Матрица вероятностей размером (n_states, n_timesteps)
        time_array: Массив временных меток размером (n_timesteps,)
        save_to_file: Флаг сохранения графика. Default: False
        filename: Имя файла для сохранения. Default: "probabilities_plot.png"
    """
    plt.figure(figsize=(12, 8))

    # Графики вероятностей для каждого состояния
    for state in range(p_matrix.shape[0]):
        plt.plot(time_array, p_matrix[state], linewidth=2)

    # График суммарной вероятности
    plt.plot(time_array, p_matrix.sum(axis=0), 'k--',
            label="Total probability", linewidth=2)

    _configure_plot("System States Probabilities", "Time (t)", "Probability")

    if save_to_file:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        return
    plt.show()


def plot_throughput(
    throughput_results: List[np.ndarray],
    time_array: np.ndarray,
    save_to_file: bool = False,
    filename: str = "throughput_plot.png"
) -> None:
    """Строит график пропускной способности.

    Args:
        throughput_results: Массив значений пропускной способности
        time_array: Массив временных меток размером (n_timesteps,)
        save_to_file: Флаг сохранения графика. Default: False
        filename: Имя файла для сохранения. Default: "throughput_plot.png"
    """
    plt.figure(figsize=(12, 8))

    # Графики пропускной способности для каждого состояния
    for state in range(len(throughput_results)):
        plt.plot(time_array, throughput_results[state], linewidth=2,
                label=f"State {state}")

    _configure_plot("System Throughput vs Impatience Rate",
                   "Impatience rate (ν)", "Throughput")

    if save_to_file:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        return
    plt.show()


def _configure_plot(title: str, xlabel: str, ylabel: str) -> None:
    """Настраивает общие параметры графиков.

    Args:
        title: Заголовок графика
        xlabel: Подпись оси X
        ylabel: Подпись оси Y
    """
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.axvline(0, color='black', linewidth=0.5)
