"""Модуль для визуализации вероятностей в системе массового обслуживания (СМО).

Содержит функцию plot_probabilities, которая отображает вероятности
состояний системы во времени, а также их суммарную вероятность.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_probabilities(p_matrix: np.ndarray, time_array: np.ndarray) -> None:
    """Строит график вероятностей для каждого состояния и их суммарной вероятности.

    Функция создает график, где каждая линия представляет вероятность определенного
    состояния во времени, а пунктирная черная линия показывает суммарную вероятность
    всех состояний.

    Args:
        p_matrix: Матрица вероятностей размером (n_states, n_timesteps), где каждая
                  строка представляет вероятности одного состояния на разных временных
                  шагах.
        time_array: Массив временных меток размером (n_timesteps,) для оси X.

    Examples:
        >>> p_matrix = np.array([[0.8, 0.6, 0.4], [0.2, 0.4, 0.6]])
        >>> time_array = np.array([0, 1, 2])
        >>> plot_probabilities(p_matrix, time_array)
        # Отобразит график с двумя линиями состояний и одной суммарной линией
    """
    plt.figure(figsize=(12, 8))

    num_states = p_matrix.shape[0]

    # Построение графиков для каждой строки матрицы p
    for index in range(num_states):
        plt.plot(time_array, p_matrix[index], linewidth=2)

    # Построение графика суммы всех вероятностей
    plt.plot(
        time_array,
        p_matrix.sum(axis=0),
        label="Total",
        color="black",
        linewidth=2,
        linestyle="--",
    )

    # Настройка графика
    plt.title("Plot of Probabilities and Total", fontsize=16, fontweight="bold")
    plt.xlabel("Time (t)", fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=12)
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.axvline(0, color="black", linewidth=0.5, linestyle="--")

    plt.show()
