import matplotlib.pyplot as plt
import numpy as np


def plot_probabilities(p_matrix: np.ndarray, time_array: np.ndarray) -> None:
    """
    Строит график вероятностей для каждой строки матрицы P и их суммы.

    :param p_matrix: матрица вероятностей P
    :param time_array: массив времени
    """
    plt.figure(figsize=(12, 8))
    
    num_states = p_matrix.shape[0]

    # Построение графиков для каждой строки матрицы p
    for i in range(num_states):
        plt.plot(time_array, p_matrix[i], label=f'p{i + 1}', linewidth=2)

    # Построение графика суммы всех вероятностей
    plt.plot(time_array, p_matrix.sum(axis=0), label='Total', color='black', linewidth=2, linestyle='--')

    # Настройка графика
    plt.title('Plot of Probabilities and Total', fontsize=16, fontweight='bold')
    plt.xlabel('Time (t)', fontsize=14)
    plt.ylabel('Values', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

    plt.show()
